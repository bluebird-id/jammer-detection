import tensorflow as tf
from ei_shared.labels import BoundingBoxLabelScore, BoundingBox

def bbox_to_segmentation(*args, validation=False):
    """
        Returns a map function that transforms a set of bounding box coordinates into a segmentation map
        indicating where the bounding box centroids are.

        The function is called by Expert Mode code, so we use args/kwargs to keep the parameters flexible
        in case of any changes.
    """

    # we should never use validation=True, but it's possible existing expert mode code does.
    if validation:
        print(support_message())
        exit(1)

    # This function was refactored to have different behavior and arguments, but is called from
    # Expert Mode. We need to catch any calls from legacy code and give the user help in porting
    # to the new version.
    if len(args) == 2:
        output_width_height: int = args[0]
        num_classes_with_background: int = args[1]
        assert isinstance(output_width_height, int), 'Expected output_width_height to be an int'
        assert isinstance(num_classes_with_background, int), 'Expected num_classes_with_background to be an int'
    else:
        print(support_message())
        exit(1)

    def mapper(x, boxes_classes):
        def get_updates(box_class):
            """
                Figures out what updates would need to be made to a fully background segmentation map
                in order to add this particular bounding box centroid.
            """
            box, label = box_class
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            bbox_label_score = BoundingBoxLabelScore(BoundingBox(x0, y0, x1, y1), label=label)
            # project bbox to output width and height
            bbox = bbox_label_score.bbox.project(output_width_height, output_width_height)
            # map centroid of box from the normalised input frame to output pixel frame
            cx, cy = tuple(bbox.centroid())
            cx, cy = tf.cast(tf.math.floor(cx), tf.int32), tf.cast(tf.math.floor(cy), tf.int32)

            # combine for compatibility with tensor_scatter_nd_update
            indices = tf.stack([cx, cy], 0)
            # add the "background" class to the one hot encoding, and be explicit about the shape for graph mode.
            label = tf.ensure_shape(tf.concat([[0.0], label], 0), (num_classes_with_background,))
            return indices, label

        # Obtain lists of all the updates we need to make
        indices, updates = tf.map_fn(get_updates, boxes_classes,
                                     fn_output_signature=(tf.TensorSpec(None, tf.int32),
                                                          tf.TensorSpec(None, tf.float32)))

        # Generate a map that represents an entire image of only background, no items
        row = tf.concat([[1.0], tf.zeros((num_classes_with_background - 1))], 0)
        flat = tf.tile(row, [output_width_height * output_width_height])
        y_map = tf.reshape(flat, (output_width_height, output_width_height, num_classes_with_background))

        # Update the map with the items
        y_map = tf.tensor_scatter_nd_update(y_map, indices=indices, updates=updates)

        if validation:
            return x, y_map, boxes_classes
        else:
            return x, y_map

    return mapper

def support_message():
    return """
---------------------------
Important: Changes required
---------------------------

We've made some changes to improve performance and work around potential errors on GPU. Please make the following updates to your Expert Mode code:

Step 1: Replace the lines where `train_segmentation_dataset` and `validation_segmentation_dataset` are defined with the following code:

    def as_segmentation(ds):
        return ds.map(dataset.bbox_to_segmentation(output_width_height, num_classes_with_background)
                      ).batch(32, drop_remainder=False).prefetch(1)
    train_segmentation_dataset = as_segmentation(train_dataset)
    validation_segmentation_dataset = as_segmentation(validation_dataset)
    validation_dataset_for_callback = validation_dataset.batch(32, drop_remainder=False).prefetch(1)

Step 2: Change the Centroid callback to use 'validation_dataset_for_callback'

    callbacks.append(metrics.CentroidScoring(validation_dataset_for_callback,
                                             output_width_height, num_classes_with_background))

After making those changes your model should train as before.

You may also return to Visual Mode, but that will reset any Expert Mode changes you've made.

Sorry for any inconvenience! Please contact support@edgeimpulse.com if you need help.
"""