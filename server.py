import grpc
from concurrent import futures
import risk_extractor_pb2
import risk_extractor_pb2_grpc
from utils import *
import os

class RiskExtractorServicer(risk_extractor_pb2_grpc.RiskExtractorServicer):
    def ExtractRisks(self, request, context):
        try:
            doc_year = decompose_query(request.query)
            doc_name = f"corp-10k-{doc_year}"  # Example mapping

            print(f"Extracting risks for document: {doc_name} for year: {doc_year}")
            
            dir_path = os.path.dirname(os.path.realpath(__file__))

            doc_path = f"/documents/{doc_name}.pdf"  
            fname = dir_path + doc_path

            risk_section = extract_risk_section(fname)
            chunks = chunk_text(risk_section)
            all_risks = []
          
            all_risks.append(extract_risks_from_chunks(chunks,  request.space_url))
            
            print(f"Logging all extracted risks: {all_risks}")

            merged = merge_risks(all_risks)
            provenanced = validate_and_add_provenance(merged, doc_name, request.query_timestamp, request.user_id, "finetuned-llm-v1.2")
            
            print(f"Logging provenanced output: {provenanced}")

            # Convert to proto response
            risks_proto = [risk_extractor_pb2.Risk(risk_categories=r['risk_categories'], risk_summary=r['risk_summary'], full_output=r["full_output"]) for r in provenanced['risks']]
            return risk_extractor_pb2.ExtractRisksResponse(
                risks=risks_proto,
                source_document=provenanced['source_document'],
                query_timestamp=provenanced['query_timestamp'],
                generation_timestamp=provenanced['generation_timestamp'],
                user_id=provenanced['user_id'],
                model_used=provenanced['model_used'],
                audit_id=provenanced['audit_id'],
                data_lineage=provenanced['data_lineage'],
                integrity_hash=provenanced['integrity_hash']
            )
        except Exception as e:
            return risk_extractor_pb2.ExtractRisksResponse(error_message=str(e))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    risk_extractor_pb2_grpc.add_RiskExtractorServicer_to_server(RiskExtractorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()