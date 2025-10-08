// Copyright (c) Meta Platforms, Inc. and affiliates.

using UnityEngine;

namespace PassthroughCameraSamples
{
    // Holds the selected Sentis model across scenes
    public static class SelectedModelRegistry
    {
        public static Unity.InferenceEngine.ModelAsset SelectedModel { get; private set; }

        public static void SetSelectedModel(Unity.InferenceEngine.ModelAsset model)
        {
            SelectedModel = model;
            if (SelectedModel != null)
            {
                PlayerPrefs.SetString("SelectedSentisModelName", SelectedModel.name);
            }
            else
            {
                PlayerPrefs.DeleteKey("SelectedSentisModelName");
            }
        }

        public static string GetLastSelectedModelName()
        {
            return PlayerPrefs.GetString("SelectedSentisModelName", string.Empty);
        }
    }
}


