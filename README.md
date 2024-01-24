# API Documentation

Welcome to the API documentation for our animal detection.

What is this API? It's a new beautiful AI model wrapped into an API to help detect a deer or a boar in an image and count them.

## Endpoints

### POST /prediction

This endpoint allows you to make a detection.

#### How to request this endpoint?

You should request this endpoint using a POST method with a JSON content. Your JSON should contain at least the "input" key with an associated value (image_url or list of image_urls).

Example:

```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"input": "http://www.pyreneanway.com/blog/wp-content/uploads/2018/11/wild-boar.jpg"}' http://localhost:5000/predict
```

It returns a list of dictionary results:

```json
    output = [{
    'img_source': image source (3 channels list),
    'img_annotated': annotated image (3 channels list),
    'number_of_detections_by_class': {'boar': nb of boars, 'deer': nb of deers},
    'boxes': [{'class_id': 0.0,
                'score': 0.9666306972503662,
                'x1': 278.8862609863281,
                'x2': 806.2059326171875,
                'y1': 0.0,
                'y2': 726.2051391601562}]
    }]
```


You can plot the annotated image using PIL:

```python
    from PIL import Image
    import numpy as np

    annotated_img = output[0]['img_annotated']

    im_array = np.array(annotated_img).astype(np.uint8)  # Convert to a BGR numpy array
    rgb_image = Image.fromarray(im_array[..., ::-1])  # Convert to an RGB PIL image
    rgb_image.show()  # Show the image
```

You can also have a list of image_urls:

```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"input": ["http://www.pyreneanway.com/blog/wp-content/uploads/2018/11/wild-boar.jpg", "https://www.mammal.org.uk/wp-content/uploads/2021/09/boar-300x300.jpg"]}' http://localhost:5000/predict
```