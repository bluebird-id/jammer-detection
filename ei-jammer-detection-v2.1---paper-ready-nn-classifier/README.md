# Local training pipeline for "Ekky Kharismadhany / Jammer Detection v2.1 - Paper Ready"

This is the local training pipeline (based on Keras / TensorFlow) for your Edge Impulse project [Ekky Kharismadhany / Jammer Detection v2.1 - Paper Ready](http://localhost:4800/studio/270913) (http://localhost:4800/studio/270913). Use it to train your model locally or run experiments. Once you're done with experimentation you can push the model back into Edge Impulse, and retrain from there.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Open a command prompt or terminal window.
3. Build the container:

    ```
    $ docker build -t custom-block-270913 .
    ```

4. Train your model:

    **macOS, Linux**

    ```
    $ docker run --rm -v $PWD:/scripts custom-block-270913 --data-directory data --out-directory out
    ```

    **Windows**

    ```
    $ docker run --rm -v "%cd%":/scripts custom-block-270913 --data-directory data --out-directory out
    ```

5. This will write your model (in TFLite, Saved Model and H5 format) to the `out/` directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

#### Adding new arguments

To add new arguments, see [Custom learning blocks > Arguments to your script](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks#arguments-to-your-script).

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects via **Create impulse > Add new learning block**.
