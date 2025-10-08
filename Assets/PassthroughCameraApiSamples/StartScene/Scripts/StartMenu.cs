// Copyright (c) Meta Platforms, Inc. and affiliates.
// Original Source code from Oculus Starter Samples (https://github.com/oculus-samples/Unity-StarterSamples)

using System;
using System.Collections.Generic;
using System.IO;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.UI;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PassthroughCameraSamples.StartScene
{
    // Create menu of all scenes included in the build.
    [MetaCodeSample("PassthroughCameraApiSamples-StartScene")]
    public class StartMenu : MonoBehaviour
    {
        public OVROverlay Overlay;
        public OVROverlay Text;
        public OVRCameraRig VrRig;

        private void Start()
        {
            var generalScenes = new List<Tuple<int, string>>();
            var passthroughScenes = new List<Tuple<int, string>>();
            var proControllerScenes = new List<Tuple<int, string>>();

            var n = UnityEngine.SceneManagement.SceneManager.sceneCountInBuildSettings;
            for (var sceneIndex = 1; sceneIndex < n; ++sceneIndex)
            {
                var path = UnityEngine.SceneManagement.SceneUtility.GetScenePathByBuildIndex(sceneIndex);

                if (path.Contains("Passthrough"))
                {
                    passthroughScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
                else if (path.Contains("TouchPro"))
                {
                    proControllerScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
                else
                {
                    generalScenes.Add(new Tuple<int, string>(sceneIndex, path));
                }
            }

            var uiBuilder = DebugUIBuilder.Instance;
            // Add model selection radios next to scene selection
            var availableModels = FindAvailableSentisModels();
            if (availableModels.Count > 0)
            {
                _ = uiBuilder.AddLabel("Sentis Models", DebugUIBuilder.DEBUG_PANE_LEFT);
                var lastSelectedName = PassthroughCameraSamples.SelectedModelRegistry.GetLastSelectedModelName();
                if (string.IsNullOrEmpty(lastSelectedName))
                {
                    // Default to first model if nothing stored
                    PassthroughCameraSamples.SelectedModelRegistry.SetSelectedModel(availableModels[0]);
                }
                foreach (var model in availableModels)
                {
                    var modelName = model != null ? model.name : "Unnamed";
                    var isDefault = !string.IsNullOrEmpty(lastSelectedName) ? modelName == lastSelectedName : false;
                    _ = uiBuilder.AddRadio(modelName, "sentismodels", (Toggle t) =>
                    {
                        if (t.isOn)
                        {
                            PassthroughCameraSamples.SelectedModelRegistry.SetSelectedModel(model);
                        }
                    }, DebugUIBuilder.DEBUG_PANE_LEFT);
                    if (isDefault)
                    {
                        PassthroughCameraSamples.SelectedModelRegistry.SetSelectedModel(model);
                    }
                }
                _ = uiBuilder.AddDivider(DebugUIBuilder.DEBUG_PANE_LEFT);
            }
            else
            {
                Debug.Log("Sentis: No .sentis models found in Editor scan or Resources. Place .sentis under Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model (Editor) or under Assets/Resources/... for runtime.");
            }
            if (passthroughScenes.Count > 0)
            {
                _ = uiBuilder.AddLabel("Passthrough Sample Scenes", DebugUIBuilder.DEBUG_PANE_LEFT);
                foreach (var scene in passthroughScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_LEFT);
                }
            }

            if (proControllerScenes.Count > 0)
            {
                _ = uiBuilder.AddLabel("Pro Controller Sample Scenes", DebugUIBuilder.DEBUG_PANE_RIGHT);
                foreach (var scene in proControllerScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_RIGHT);
                }
            }

            _ = uiBuilder.AddLabel("Press ☰ at any time to return to scene selection", DebugUIBuilder.DEBUG_PANE_CENTER);
            if (generalScenes.Count > 0)
            {
                _ = uiBuilder.AddDivider(DebugUIBuilder.DEBUG_PANE_CENTER);
                _ = uiBuilder.AddLabel("Sample Scenes", DebugUIBuilder.DEBUG_PANE_CENTER);
                foreach (var scene in generalScenes)
                {
                    _ = uiBuilder.AddButton(Path.GetFileNameWithoutExtension(scene.Item2), () => LoadScene(scene.Item1), -1, DebugUIBuilder.DEBUG_PANE_CENTER);
                }
            }

            uiBuilder.Show();
        }

        private static List<Unity.InferenceEngine.ModelAsset> FindAvailableSentisModels()
        {
            var models = new List<Unity.InferenceEngine.ModelAsset>();

#if UNITY_EDITOR
            // Editor-only: robust scan the folder for any asset paths containing ".sentis"
            var searchFolders = new[] { "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model" };
            var allGuids = AssetDatabase.FindAssets(string.Empty, searchFolders);
            foreach (var guid in allGuids)
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                if (!string.IsNullOrEmpty(path) && path.ToLowerInvariant().Contains(".sentis"))
                {
                    var asset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(path);
                    if (asset != null)
                    {
                        models.Add(asset);
                    }
                }
            }
            Debug.Log($"Sentis: Editor scan found {models.Count} model(s).");
#else
            // Runtime: attempt to load from Resources if user placed models there
            var runtimeModels = Resources.LoadAll<Unity.InferenceEngine.ModelAsset>(
                "PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model");
            if (runtimeModels != null && runtimeModels.Length > 0)
            {
                models.AddRange(runtimeModels);
            }
            // Also allow any location under Resources root
            if (models.Count == 0)
            {
                var anyModels = Resources.LoadAll<Unity.InferenceEngine.ModelAsset>(string.Empty);
                if (anyModels != null && anyModels.Length > 0)
                {
                    models.AddRange(anyModels);
                }
            }
#endif

            return models;
        }

        private void LoadScene(int idx)
        {
            DebugUIBuilder.Instance.Hide();
            Debug.Log("Load scene: " + idx);
            UnityEngine.SceneManagement.SceneManager.LoadScene(idx);
        }
    }
}
