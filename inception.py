import numpy as np
import tensorflow as tf
import download
from cache import cache
import os
import sys

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# Directory to store the downloaded data.
data_dir = "inception/"

# File containing the mappings between class-number and uid. (Downloaded)
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"

# File containing the mappings between uid and string. (Downloaded)
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "classify_image_graph_def.pb"


def maybe_download():
    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


class NameLookup:
    def __init__(self):
        # Mappings between uid, cls and name are dicts, where insertions and
        # lookup have O(1) time-usage on average, but may be O(n) in worst case.
        self._uid_to_cls = {}   # Map from uid to cls.
        self._uid_to_name = {}  # Map from uid to name.
        self._cls_to_uid = {}   # Map from cls to uid.

        # Read the uid-to-name mappings from file.
        path = os.path.join(data_dir, path_uid_to_name)
        with open(file=path, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # Remove newlines.
                line = line.replace("\n", "")

                # Split the line on tabs.
                elements = line.split("\t")

                # Get the uid.
                uid = elements[0]

                # Get the class-name.
                name = elements[1]

                # Insert into the lookup-dict.
                self._uid_to_name[uid] = name

        # Read the uid-to-cls mappings from file.
        path = os.path.join(data_dir, path_uid_to_cls)
        with open(file=path, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # We assume the file is in the proper format,
                # so the following lines come in pairs. Other lines are ignored.

                if line.startswith("  target_class: "):
                    # This line must be the class-number as an integer.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the class-number as an integer.
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    # This line must be the uid as a string.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the uid as a string e.g. "n01494475"
                    uid = elements[1]

                    # Remove the enclosing "" from the string.
                    uid = uid[1:-2]

                    # Insert into the lookup-dicts for both ways between uid and cls.
                    self._uid_to_cls[uid] = cls
                    self._cls_to_uid[cls] = uid

    def uid_to_cls(self, uid):
        return self._uid_to_cls[uid]

    def uid_to_name(self, uid, only_first_name=False):
        name = self._uid_to_name[uid]

        # Only use the first name in the list?
        if only_first_name:
            name = name.split(",")[0]

        return name

    def cls_to_name(self, cls, only_first_name=False):
        # Lookup the uid from the cls.
        uid = self._cls_to_uid[cls]

        # Lookup the name from the uid.
        name = self.uid_to_name(uid=uid, only_first_name=only_first_name)

        return name


class Inception:
    # Name of the tensor for feeding the input image as jpeg.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"

    # Name of the tensor for feeding the decoded input image.
    # Use this for feeding images in other formats than jpeg.
    tensor_name_input_image = "DecodeJpeg:0"

    # Name of the tensor for the resized input image.
    # This is used to retrieve the image after it has been resized.
    tensor_name_resized_image = "ResizeBilinear:0"

    # Name of the tensor for the output of the softmax-classifier.
    # This is used for classifying images with the Inception model.
    tensor_name_softmax = "softmax:0"

    # Name of the tensor for the unscaled outputs of the softmax-classifier (aka. logits).
    tensor_name_softmax_logits = "softmax/logits:0"

    # Name of the tensor for the output of the Inception model.
    # This is used for Transfer Learning.
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):
        self.name_lookup = NameLookup()

        self.graph = tf.Graph()
        with self.graph.as_default():

            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:

                graph_def = tf.GraphDef()

                graph_def.ParseFromString(file.read())

                tf.import_graph_def(graph_def, name='')

        self.y_pred = self.graph.get_tensor_by_name(self.tensor_name_softmax)

        self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)

        self.resized_image = self.graph.get_tensor_by_name(self.tensor_name_resized_image)

        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        self.transfer_len = self.transfer_layer.get_shape()[3]

        self.session = tf.Session(graph=self.graph)

    def close(self):
        self.session.close()

    def _write_summary(self, logdir='summary/'):
        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def _create_feed_dict(self, image_path=None, image=None):
        feed_dict = {self.tensor_name_input_image: image}
        return feed_dict

    def classify(self, image_path=None, image=None):

        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        pred = self.session.run(self.y_pred, feed_dict=feed_dict)

        pred = np.squeeze(pred)

        return pred

    def get_resized_image(self, image_path=None, image=None):

        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)

        resized_image = resized_image.squeeze(axis=0)

        resized_image = resized_image.astype(float) / 255.0

        return resized_image

    def print_scores(self, pred, k=10, only_first_name=True):

        idx = pred.argsort()

        top_k = idx[-k:]

        for cls in reversed(top_k):
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)

            score = pred[cls]

            print("{0:>6.2%} : {1}".format(score, name))

    def transfer_values(self, image_path=None, image=None):

        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        transfer_values = np.squeeze(transfer_values)

        return transfer_values


def process_images(fn, images=None, image_paths=None):

    num_images = len(images)

    result = [None] * num_images

    for i in range(num_images):

        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        sys.stdout.write(msg)
        sys.stdout.flush()

        result[i] = fn(image_path=image_paths, image=images[i])

    result = np.array(result)

    return result


def transfer_values_cache(cache_path, model, images=None, image_paths=None):

    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)

    transfer_values = cache(cache_path=cache_path, fn=fn)

    return transfer_values






"""
if __name__ == '__main__':
    print(tf.__version__)

    # Download Inception model if not already done.
    maybe_download()

    # Load the Inception model so it is ready for classifying images.
    model = Inception()

    # Path for a jpeg-image that is included in the downloaded data.
    image_path = os.path.join(data_dir, 'cropped_panda.jpg')

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10)

    # Close the TensorFlow session.
    model.close()
"""