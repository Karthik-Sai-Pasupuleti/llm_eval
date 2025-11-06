import mlflow
import os
import shutil

# Path to your mlruns directory (default: where MLflow stores runs)
MLRUNS_PATH = "mlruns"

# Get list of all experiments
experiments = mlflow.get_experiment()

for exp in experiments:
    exp_id = exp.experiment_id
    exp_name = exp.name
    artifact_location = exp.artifact_location

    print(f"Deleting experiment: {exp_name} (ID: {exp_id})")

    # 1. Delete experiment from MLflow tracking
    try:
        mlflow.delete_experiment(exp_id)
        print(f"✅ Experiment {exp_name} deleted from tracking.")
    except Exception as e:
        print(f"❌ Failed to delete experiment {exp_name}: {e}")

    # 2. Delete artifact directory (local or remote)
    # For local files (file:// or plain path)
    if artifact_location.startswith("file:"):
        artifact_path = artifact_location.replace("file://", "")
    else:
        artifact_path = artifact_location

    if os.path.exists(artifact_path):
        try:
            shutil.rmtree(artifact_path)
            print(f"🗑️  Artifacts deleted: {artifact_path}")
        except Exception as e:
            print(f"⚠️ Could not delete artifacts for {exp_name}: {e}")
    else:
        print(f"⚠️ Artifact path not found: {artifact_path}")

print("✅ All deletions completed.")
