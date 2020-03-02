from config import get_config
from Learner import face_learner

if __name__ == '__main__':
    conf = get_config(training=True)
    learner = face_learner(conf)
    learner.train(conf)