import importlib
import sys

from learner import face_learner2


if __name__ == '__main__':
    conf_file = sys.argv[1]
    conf = importlib.import_module(conf_file.strip(
        '.py').replace('/', '.')).get_config()
    learner = face_learner2(conf, 'Test')
    learner.test()
