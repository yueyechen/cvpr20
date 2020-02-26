from config import get_config
from learner import face_learner


if __name__ == '__main__':
    conf = get_config(training=False)
    learner = face_learner(conf, inference=True)
    learner.test(conf)

