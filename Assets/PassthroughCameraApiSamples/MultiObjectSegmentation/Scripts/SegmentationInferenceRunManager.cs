// Copyright (c) Meta Platforms, Inc. and affiliates.

using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectSegmentation
{
    // Thin wrapper to keep scene/components distinct from detection sample.
    // Uses the same runtime behavior as detection for now (draws class-colored boxes).
    // Swap in a segmentation-aware UI manager later to render masks when available.
    public class SegmentationInferenceRunManager : PassthroughCameraSamples.MultiObjectDetection.SentisInferenceRunManager
    {
    }
}


