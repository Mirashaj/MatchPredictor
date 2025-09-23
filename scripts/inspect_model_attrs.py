import joblib, os, json
models_dir = 'models'
print('Models in', models_dir)
for fn in sorted(os.listdir(models_dir)):
    path = os.path.join(models_dir, fn)
    if fn.endswith('.joblib') or fn.endswith('.json'):
        print('\n---', fn)
    if fn.endswith('.joblib'):
        try:
            m = joblib.load(path)
            print('type:', type(m))
            if hasattr(m, 'feature_names_in_'):
                print(' feature_names_in_ len:', len(m.feature_names_in_))
            if hasattr(m, 'n_features_in_'):
                print(' n_features_in_:', getattr(m, 'n_features_in_'))
            # for sklearn pipelines or ColumnTransformer, try to introspect named features
            if hasattr(m, 'estimators_'):
                print(' has estimators_ (ensemble)')
        except Exception as e:
            print(' load error:', e)
    if fn.endswith('feature_encoders_with_elo.joblib') or fn.endswith('feature_encoders.joblib'):
        try:
            enc = joblib.load(path)
            print(' encoders keys:', list(enc.keys()))
            for k,v in enc.items():
                try:
                    classes = getattr(v, 'classes_', None)
                    if classes is not None:
                        print(f"  encoder {k} classes: {len(classes)}")
                except Exception:
                    pass
        except Exception as e:
            print(' enc load error:', e)
    if fn.endswith('scaler_with_elo.joblib') or fn.endswith('scaler.joblib'):
        try:
            s = joblib.load(path)
            print(' scaler type:', type(s))
            if hasattr(s, 'feature_names_in_'):
                print(' scaler.feature_names_in_ len:', len(s.feature_names_in_))
                print(' scaler.feature_names_in_ sample:', list(s.feature_names_in_)[:20])
        except Exception as e:
            print(' scaler load error:', e)
    if fn.endswith('.json'):
        try:
            with open(path,'r') as f:
                print(' json size:', len(f.read()))
        except Exception as e:
            print(' json load error:', e)
print('\nDone')
