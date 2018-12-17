import logging
from tensorboard import default
from tensorboard import program

class TensorBoardTool:

    def __init__(self, tensorboard_dir='./tensorboard'):
        self.dir_path = tensorboard_dir

    def run(self):
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)