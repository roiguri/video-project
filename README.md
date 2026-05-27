# Video Processing Project

Final project for Tel Aviv University's Video Processing course (Spring 2025).

A four-stage CV pipeline implemented from scratch with Python and OpenCV:

1. **Stabilization** — feature-based homography stabilization with smoothing
2. **Background subtraction** — KDE-based foreground extraction
3. **Image matting** — alpha compositing with distance-transform-driven trimaps
4. **Person tracking** — particle filter

Each stage can be toggled in `Code/main.py`.

## Run

```bash
pip install -r requirements.txt
cd Code && python main.py
```

Inputs are read from `Inputs/`; outputs are written to `Outputs/`.

## Status

Coursework project. Not maintained — provided as-is.
