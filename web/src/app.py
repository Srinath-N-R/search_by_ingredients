from flask import Flask, request, jsonify, render_template
from opensearchpy import OpenSearch
from decouple import config
from keto_helpers import is_keto
from vegan_helpers import is_vegan
from time import sleep
import sys
import logging
import time

# Configure logging - keep it simple
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'  # Simplified format
)
# Silence noisy loggers
logging.getLogger('opensearchpy').setLevel(logging.ERROR)  # Only show errors
logging.getLogger('urllib3').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('opensearch').setLevel(logging.ERROR)    # Only show errors
logger = logging.getLogger(__name__)

app = Flask(__name__)


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """Wait for OpenSearch to be ready"""
    print("Waiting for OpenSearch to be ready...")  # Simple status message
    for i in range(max_retries):
        try:
            if client.ping():
                # Simple success message
                print("Successfully connected to OpenSearch!")
                return True
        except Exception as e:
            # Log connection attempts at debug level
            logger.debug(f"Connection attempt {i+1}/{max_retries} failed.")
        sleep(retry_interval)
    logger.error("Failed to connect to OpenSearch after maximum retries")
    return False


def init_opensearch():
    """Initialize OpenSearch client and wait for it to be ready"""
    client = OpenSearch(
        hosts=[config('OPENSEARCH_URL', 'http://localhost:9200')],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    if not wait_for_opensearch(client):
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    try:
        # Load all ingredients once OpenSearch is ready
        response = client.search(index="ingredients", body={
                                 "query": {"match_all": {}}}, size=10000)
        ingredients = [hit["_source"]["ingredients"]
                       for hit in response["hits"]["hits"]]
        # Simple status message
        print(f"Successfully loaded {len(ingredients)} ingredients")
        return client, ingredients
    except Exception as e:
        logger.error(f"Error initializing OpenSearch: {str(e)}")
        sys.exit(1)


logger.info("Starting application initialization...")
# Initialize OpenSearch and load ingredients
client, ingredients = init_opensearch()
logger.info("Application initialization completed successfully")


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    q = request.args.get("q", "").strip()
    results = [{"id": id_, "text": txt_}
               for id_, txt_ in enumerate(ingredients) if q in txt_]
    results = sorted(results, key=lambda x: len(x["text"]))
    return jsonify({"results": results})


def build_query(ing_text: str, keto: bool, vegan: bool) -> dict:
    """
    Build an OpenSearch bool query based on ing_text and keto, vegan filters.
    """
    bool_q = {"filter": []}

    if ing_text:
        bool_q["must"] = [{
            "match": {
                "ingredients": {
                    "query": ing_text,
                    "fuzziness": "AUTO"
                }
            }
        }]
    else:
        bool_q["must"] = [{"match_all": {}}]

    if keto:
        bool_q["filter"].append({"term": {"is_keto": True}})
    if vegan:
        bool_q["filter"].append({"term": {"is_vegan": True}})

    return {"query": {"bool": bool_q}, "size": 12}


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    start_time = time.time()

    q_txt   = (request.args.get('q') or '').strip()
    keto_q  = request.args.get('keto')  == '1'
    vegan_q = request.args.get('vegan') == '1'

    if not q_txt:
        return jsonify({'error': 'Please provide an ingredient name'}), 400

    parse_start = time.time()

    logger.info(f"Parsing time: {time.time() - parse_start:.3f} seconds")

    body = build_query(q_txt, keto_q, vegan_q)
    
    search_start = time.time()
    try:
        # Execute the search
        response = client.search(index="recipes", body=body, size=12)
        logger.info(f"OpenSearch query time: {time.time() - search_start:.3f} seconds")

        # Format the results
        hits = response['hits']['hits']
        results = [{
            "title": hit["_source"]["title"],
            "description": hit["_source"].get("description", ""),
            "ingredients": hit["_source"]["ingredients"],
            "instructions": hit["_source"].get("instructions", ""),
            "photo_url": hit["_source"].get("photo_url", ""),
            "keto": hit["_source"].get("is_keto",  False),
            "vegan": hit["_source"].get("is_vegan", False),
            "score": hit["_score"]
        } for hit in hits]

        logger.info(f"Total /search endpoint time: {time.time() - start_time:.3f} seconds")

        return jsonify({
            'total': response['hits']['total']['value'],
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
