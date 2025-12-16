from src.steps.ingest import ingest
from src.steps.featurize import featurize
from src.steps.split import split
from Spotify_Analysis.src.steps.models import build_models
from src.steps.evaluate import evaluate_model
from src.steps.export import export_artifacts

def main():
    
    df = ingest()
    df = featurize(df)
    
    X_train, X_test, y_train, y_test = split(df)
    
    models = build_models()
    
    all_results = {}
    best_name = None
    best_model = None
    best_score = -1.0
    
    for name, model in models.items():
        res = evaluate_model(model, X_train, y_train, X_test, y_test)
        all_results[name] = res

        score = res["cv_mean"]["roc_auc"]
        print(f"{name}: CV ROC-AUC={score:.3f} | Test ROC-AUC={res['test_roc_auc']:.3f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    
    print(f"\n Best model by CV ROC-AUC: {best_name} ({best_score:.3f})")
    
    best_model.fit(X_train, y_train)
    
    export_artifacts(best_name, best_model, all_results)
    print("Saved model to artifacts/models/model.joblib")
    print("Saved metrics to artifacts/reports/metrics.json")
    

if __name__ == "__main__":
    main()