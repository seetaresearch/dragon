import os
import time
from multiprocessing import Process
try:
    from flask import Flask, render_template, make_response, jsonify, request
except ImportError as e: pass
from six.moves import range as xrange

class DragonBoard(Process):
    def __init__(self, log_dir='', port=5000, max_display=1000):
        super(DragonBoard, self).__init__()
        self.daemon = True
        import os
        import sys
        if log_dir == '':
            log_dir = os.path.join(os.path.abspath(os.curdir),
                                    'logs').replace('\\', '/')

        self.config = {'exec_py': sys.argv[0],
                       'log_dir': log_dir,
                       'port': port,
                       'max_display': max_display}
        def cleanup():
            from dragon.config import logger
            logger.info('Terminating DragonBoard......')
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def run(self):
        app = Flask(__name__, static_url_path='')
        app.jinja_env.variable_start_string = '%%'
        app.jinja_env.variable_end_string = '%%'
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/workspace')
        def workspace():
            return make_response(jsonify(self.config))

        @app.route('/events/list_scalars', methods=['GET', 'POST'])
        def list_sclars():
            def get_mtime(file):
                t = time.localtime(os.stat(file).st_mtime)
                return '%d:%d:%d @ %d/%d/%d  ' % (t[3], t[4], t[5], t[1], t[2], t[0])
            scalar_dir = os.path.join(self.config['log_dir'], 'scalar')
            files = os.listdir(scalar_dir)
            ret = {}
            for file in files:
                ret[file.split('.')[0]] = get_mtime(os.path.join(scalar_dir, file))
            return make_response(jsonify(ret))

        @app.route('/events/get_scalar', methods=['GET', 'POST'])
        def get_scalar():
            scalar_dir = os.path.join(self.config['log_dir'], 'scalar')
            require_file = os.path.join(scalar_dir,
                                 request.values.get('scalar') + '.txt')
            if not os.path.exists(require_file): return
            sclar = {}
            inds = {}; cur_idx = 0
            with open(require_file, 'r') as f:
                for line in f:
                    elements = line.split()
                    sclar[elements[0]] = elements[1]
                    inds[cur_idx] = elements[0]
                    cur_idx += 1

            if len(sclar) > self.config['max_display']:
                sample_scalar = {}
                stride = len(sclar) // self.config['max_display']
                for i in xrange(0, cur_idx, stride):
                    sample_scalar[inds[i]] = sclar[inds[i]]
                return make_response(jsonify(sample_scalar))
            else: return make_response(jsonify(sclar))

        app.run(host='0.0.0.0', port=self.config['port'])
