# User Manual

## Purpose

Apple-Fruit-Disease-Detection is an experimental image-classification aid for
four apple image categories: Apple Blotch, Healthy Apple, Apple Rot, and Apple
Scab.

## Use the application

1. Start the app with `streamlit run app.py`.
2. Open **Disease Prediction**.
3. Drag or browse for a JPG, JPEG, or PNG image under 10 MB.
4. Review the image preview, predicted class, confidence, and probability
   chart.
5. Open **Prediction History** to review local metadata or clear it.
6. Open **About Model** for model details and offline Phase 7 metrics.

## Interpreting results

Confidence is the model's probability score among the four supported classes;
it is not a guaranteed diagnosis. Seek qualified agricultural advice before
making treatment or crop-management decisions. The application does not
identify disease regions or draw bounding boxes.

## Privacy and limitations

Uploaded image bytes are processed in memory and are not saved by default.
Prediction metadata includes timestamp, sanitized filename, class, confidence,
and device in a per-session local SQLite file. Each Streamlit session has
separate history and clear-history scope.
