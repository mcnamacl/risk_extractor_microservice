import grpc
import risk_extractor_pb2
import risk_extractor_pb2_grpc
import time
import json

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = risk_extractor_pb2_grpc.RiskExtractorStub(channel)
        request = risk_extractor_pb2.ExtractRisksRequest(
            query="What are the risks in the report from 2022?",
            user_id="user123",
            query_timestamp=int(time.time()),
            space_url="mcnamacl/tinyllama-inference"
        )
        response = stub.ExtractRisks(request)
        print (f"Logging response: {response}")
        if response.error_message:
            print(f"Error: {response.error_message}")
        else:
            # Convert to JSON for output
            output = {
                "risks": [{"risk_categories": list(r.risk_categories), "risk_summary": r.risk_summary} for r in response.risks],
                "source_document": response.source_document,
            }
            print(json.dumps(output, indent=2))

            full_output = {
                "risks": output["risks"],
                "full_output" : [{"full_output": r.full_output} for r in response.risks],
                "source_document": response.source_document,
                "query_timestamp": response.query_timestamp,
                "generation_timestamp": response.generation_timestamp,
                "user_id": response.user_id,
                "model_used": response.model_used,
                "audit_id": response.audit_id,
                "data_lineage": response.data_lineage,
                "integrity_hash": response.integrity_hash
            }

            with ("output.json", "w+") as f:
                json.dump(full_output, f, indent=2)

if __name__ == '__main__':
    run()