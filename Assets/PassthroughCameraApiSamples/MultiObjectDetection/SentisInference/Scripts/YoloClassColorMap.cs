// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    /// <summary>
    /// Provides a semantic grouping and color mapping for YOLO classes (COCO80).
    /// Colors are assigned per semantic group to keep similar classes visually similar.
    /// </summary>
    public static class YoloClassColorMap
    {
        // Base colors per semantic group (high-contrast, color-blind friendly leaning)
        private static readonly Dictionary<string, Color> s_groupBaseColors = new()
        {
            { "person", new Color(0.95f, 0.35f, 0.35f) }, // red-ish
            { "vehicle", new Color(0.95f, 0.75f, 0.25f) }, // yellow/orange
            { "traffic", new Color(1.00f, 0.55f, 0.00f) }, // orange
            { "animal", new Color(0.30f, 0.70f, 0.40f) }, // green
            { "bag", new Color(0.45f, 0.60f, 0.85f) }, // blue
            { "sports", new Color(0.55f, 0.45f, 0.85f) }, // purple
            { "kitchen", new Color(0.25f, 0.75f, 0.75f) }, // teal
            { "food", new Color(0.95f, 0.55f, 0.65f) }, // pink
            { "furniture", new Color(0.60f, 0.60f, 0.60f) }, // gray
            { "appliance", new Color(0.35f, 0.75f, 0.95f) }, // sky
            { "electronics", new Color(0.20f, 0.60f, 0.80f) }, // blue
            { "bath", new Color(0.30f, 0.80f, 0.60f) }, // mint
            { "plant", new Color(0.40f, 0.75f, 0.40f) }, // plant green
            { "book_misc", new Color(0.70f, 0.70f, 0.30f) } // olive
        };

        // Class to group mapping for COCO80 classes used in SentisYoloClasses.txt
        private static readonly Dictionary<string, string> s_classToGroup = new()
        {
            // people
            { "person", "person" },

            // vehicles
            { "bicycle", "vehicle" }, { "car", "vehicle" }, { "motorbike", "vehicle" }, { "aeroplane", "vehicle" },
            { "bus", "vehicle" }, { "train", "vehicle" }, { "truck", "vehicle" }, { "boat", "vehicle" },

            // traffic and street objects
            { "traffic_light", "traffic" }, { "fire_hydrant", "traffic" }, { "stop_sign", "traffic" }, { "parking_meter", "traffic" },

            // outdoor furniture
            { "bench", "furniture" },

            // animals
            { "bird", "animal" }, { "cat", "animal" }, { "dog", "animal" }, { "horse", "animal" }, { "sheep", "animal" },
            { "cow", "animal" }, { "elephant", "animal" }, { "bear", "animal" }, { "zebra", "animal" }, { "giraffe", "animal" },

            // accessories / bags
            { "backpack", "bag" }, { "umbrella", "bag" }, { "handbag", "bag" }, { "tie", "bag" }, { "suitcase", "bag" },

            // sports
            { "frisbee", "sports" }, { "skis", "sports" }, { "snowboard", "sports" }, { "sports_ball", "sports" }, { "kite", "sports" },
            { "baseball_bat", "sports" }, { "baseball_glove", "sports" }, { "skateboard", "sports" }, { "surfboard", "sports" }, { "tennis_racket", "sports" },

            // kitchen
            { "bottle", "kitchen" }, { "wine_glass", "kitchen" }, { "cup", "kitchen" }, { "fork", "kitchen" }, { "knife", "kitchen" },
            { "spoon", "kitchen" }, { "bowl", "kitchen" },

            // food
            { "banana", "food" }, { "apple", "food" }, { "sandwich", "food" }, { "orange", "food" }, { "broccoli", "food" },
            { "carrot", "food" }, { "hot_dog", "food" }, { "pizza", "food" }, { "donut", "food" }, { "cake", "food" },

            // furniture
            { "chair", "furniture" }, { "sofa", "furniture" }, { "bed", "furniture" }, { "diningtable", "furniture" },

            // plant
            { "pottedplant", "plant" },

            // bath
            { "toilet", "bath" },

            // electronics
            { "tvmonitor", "electronics" }, { "laptop", "electronics" }, { "mouse", "electronics" }, { "remote", "electronics" },
            { "keyboard", "electronics" }, { "cell_phone", "electronics" },

            // appliances
            { "microwave", "appliance" }, { "oven", "appliance" }, { "toaster", "appliance" }, { "sink", "appliance" }, { "refrigerator", "appliance" },

            // misc
            { "book", "book_misc" }, { "clock", "book_misc" }, { "vase", "book_misc" }, { "scissors", "book_misc" },
            { "teddy_bear", "book_misc" }, { "hair_drier", "book_misc" }, { "toothbrush", "book_misc" }
        };

        private static readonly Color s_fallbackColor = new(0.95f, 0.95f, 0.95f);

        public static Color GetColorForClass(string className)
        {
            if (string.IsNullOrEmpty(className))
            {
                return s_fallbackColor;
            }

            // normalization: class names are stored with underscores in code
            var normalized = className.Replace(" ", "_");

            if (s_classToGroup.TryGetValue(normalized, out var group))
            {
                if (s_groupBaseColors.TryGetValue(group, out var baseColor))
                {
                    // Slight deterministic variation within group based on class hash
                    var t = Mathf.Abs(normalized.GetHashCode() % 1000) / 1000.0f; // 0..1
                    var valueJitter = Mathf.Lerp(0.9f, 1.1f, t);
                    var r = Mathf.Clamp01(baseColor.r * valueJitter);
                    var g = Mathf.Clamp01(baseColor.g * valueJitter);
                    var b = Mathf.Clamp01(baseColor.b * valueJitter);
                    return new Color(r, g, b, 1.0f);
                }
            }

            return s_fallbackColor;
        }
    }
}


