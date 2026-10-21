from __future__ import annotations
import argparse, json, mimetypes, socket, sys, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

try:
    import joblib
    import numpy as np
    from catboost import CatBoostRegressor
except ModuleNotFoundError as exc:
    print(f"[ERROR] Missing Python package: {exc.name}")
    print("Run start_windows.bat. The launcher creates a Python 3.11 virtual environment and installs dependencies.")
    raise SystemExit(2)

if sys.version_info[:2] != (3, 11):
    print(f"[ERROR] FRP Bond Predictor v1.0 requires Python 3.11.x. Current: {sys.version.split()[0]}")
    raise SystemExit(3)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODELS = ROOT / "models"
FEATURES = ['Surface','Rh_d','Rs_d','d','c_d','lb_d','Ar','as','Rr','Type','Ef','ffu','fc']
RANGES = {
    'Rh_d':[0.01,0.17], 'Rs_d':[0.47,3.49], 'd':[5.5,16.0], 'c_d':[1.25,13.13],
    'lb_d':[4.0,7.0], 'Ar':[4.57,111.79], 'as':[0.47,13.92], 'Rr':[0.01,0.28],
    'Ef':[21.3,200.0], 'ffu':[396.0,1801.6], 'fc':[23.3,67.1], 'tau':[5.10,30.95]
}
REPORTED = {
    'CatBoost': {'R2':0.841853,'RMSE':2.196218,'MAE':1.649743,'MAPE':11.56876},
    'XGBoost': {'R2':0.836434,'RMSE':2.233527,'MAE':1.650063,'MAPE':11.70154},
    'GBDT': {'R2':0.831853,'RMSE':2.264588,'MAE':1.760162,'MAPE':12.86751},
    'Random Forest': {'R2':0.812891,'RMSE':2.388864,'MAE':1.849388,'MAPE':13.3012},
    'BPNN': {'R2':0.810294,'RMSE':2.405386,'MAE':1.824587,'MAPE':12.83238},
    'SVR': {'R2':0.790543,'RMSE':2.527504,'MAE':1.865332,'MAPE':12.70128},
    'AdaBoost': {'R2':0.695248,'RMSE':3.048723,'MAE':2.431504,'MAPE':17.59481},
    'MLR': {'R2':0.341504,'RMSE':4.481476,'MAE':3.411779,'MAPE':23.70651},
    'LASSO': {'R2':0.325327,'RMSE':4.536188,'MAE':3.348338,'MAPE':23.37707}
}

def verify_model_files():
    required = [
        MODELS/'catboost.cbm', MODELS/'xgboost.joblib', MODELS/'gbdt.joblib',
        MODELS/'random_forest.joblib', MODELS/'bpnn.joblib', MODELS/'svr.joblib',
        MODELS/'adaboost.joblib', MODELS/'mlr.joblib', MODELS/'lasso.joblib',
        MODELS/'model_meta.json'
    ]
    miss = [str(p.name) for p in required if not p.exists()]
    if miss:
        raise SystemExit("[ERROR] Model files are missing: " + ", ".join(miss) + "\nRun train_models.py or start_windows.bat.")

verify_model_files()

class ModelStore:
    def __init__(self):
        self.models = {}
        cat = CatBoostRegressor()
        cat.load_model(str(MODELS / 'catboost.cbm'))
        self.models['CatBoost'] = cat
        mapping = {
            'XGBoost':'xgboost.joblib','GBDT':'gbdt.joblib','Random Forest':'random_forest.joblib',
            'BPNN':'bpnn.joblib','SVR':'svr.joblib','AdaBoost':'adaboost.joblib',
            'MLR':'mlr.joblib','LASSO':'lasso.joblib'
        }
        for name, fn in mapping.items():
            self.models[name] = joblib.load(MODELS / fn)
        self.meta = json.loads((MODELS/'model_meta.json').read_text(encoding='utf-8'))

    def predict(self, model_name: str, values: dict) -> float:
        if model_name not in self.models:
            raise ValueError(f'Unknown model: {model_name}')
        row = [float(values[k]) for k in FEATURES]
        if model_name == 'CatBoost':
            row2 = row.copy()
            row2[0] = str(int(round(row2[0])))
            row2[9] = str(int(round(row2[9])))
            return float(self.models[model_name].predict([row2])[0])
        return float(self.models[model_name].predict(np.asarray([row], dtype=float))[0])

STORE = ModelStore()

def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('1.1.1.1', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return '127.0.0.1'

class Handler(BaseHTTPRequestHandler):
    server_version = 'FRP-BondPredictor/1.0'

    def _json(self, obj, code=200):
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text/') else ''))
        self.send_header('Cache-Control','no-cache')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/api/health':
            return self._json({'ok':True,'version':'1.0.0','python':sys.version.split()[0]})
        if p == '/api/info':
            ip = lan_ip()
            return self._json({
                'app':'FRP bar-concrete bond strength predictor','version':'1.0.0',
                'models':list(STORE.models.keys()),
                'features':FEATURES,'ranges':RANGES,'reported_metrics':REPORTED,
                'model_meta':STORE.meta,
                'mobile_url':f'http://{ip}:8765/mobile',
                'desktop_url':'http://127.0.0.1:8765/',
                'training_basis':'195 experimental records reconstructed from Appendix A of the supplied thesis PDF.',
                'paper_dataset_note':'The thesis text reports 193 valid samples after outlier removal; the final cleaned 193-row CSV is still required for exact paper replication.'
            })
        if p in ('/','/index.html'):
            ua = self.headers.get('User-Agent','').lower()
            if any(k in ua for k in ('iphone','android','mobile')):
                return self._file(STATIC/'mobile.html')
            return self._file(STATIC/'index.html')
        if p == '/mobile':
            return self._file(STATIC/'mobile.html')
        if p == '/desktop':
            return self._file(STATIC/'index.html')
        safe = (STATIC / p.lstrip('/')).resolve()
        if STATIC.resolve() not in safe.parents and safe != STATIC.resolve():
            self.send_error(403)
            return
        return self._file(safe)

    def do_POST(self):
        if urlparse(self.path).path != '/api/predict':
            return self._json({'error':'not found'},404)
        try:
            n = int(self.headers.get('Content-Length','0'))
            body = json.loads(self.rfile.read(n).decode('utf-8'))
            model = body.get('model','CatBoost')
            values = body.get('features',{})
            missing = [k for k in FEATURES if k not in values]
            if missing:
                raise ValueError('Missing features: ' + ', '.join(missing))
            pred = STORE.predict(model, values)
            out = []
            for k,r in RANGES.items():
                if k == 'tau' or k not in values:
                    continue
                v=float(values[k])
                if not (r[0] <= v <= r[1]):
                    out.append({'feature':k,'value':v,'range':r})
            return self._json({
                'prediction_MPa':pred,
                'model':model,
                'out_of_domain':out,
                'is_extrapolation':bool(out),
                'reported_cv_metrics':REPORTED[model],
                'model_training_rows':STORE.meta.get('rows'),
                'model_python':STORE.meta.get('python')
            })
        except Exception as e:
            return self._json({'error':str(e)},400)

    def log_message(self, fmt, *args):
        print('[FRP Bond Predictor]', fmt % args)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--host',default='0.0.0.0')
    ap.add_argument('--port',type=int,default=8765)
    ap.add_argument('--no-browser',action='store_true')
    args=ap.parse_args()

    srv=ThreadingHTTPServer((args.host,args.port),Handler)
    local=f'http://127.0.0.1:{args.port}'
    mobile=f'http://{lan_ip()}:{args.port}/mobile'
    print()
    print("="*70)
    print(" FRP bond-strength predictor v1.0.0 - Python 3.11 local web application")
    print(" Desktop :", local)
    print(" Mobile  :", mobile)
    print(" Phone and PC must be connected to the same LAN/Wi-Fi.")
    print(" If Windows Firewall prompts, allow access on Private networks.")
    print("="*70)
    print()

    if not args.no_browser:
        threading.Timer(1.0, lambda:webbrowser.open(local)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

if __name__=='__main__':
    main()
