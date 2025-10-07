// Copyright (c) Meta Platforms, Inc. and affiliates.

using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectSegmentation
{
    // For now, reuse the detection UI which draws boxes around segmented objects.
    // Later, this can be replaced with a shader-based overlay to render masks.
    public class SegmentationInferenceUiManager : PassthroughCameraSamples.MultiObjectDetection.SentisInferenceUiManager
    {
    }
}


