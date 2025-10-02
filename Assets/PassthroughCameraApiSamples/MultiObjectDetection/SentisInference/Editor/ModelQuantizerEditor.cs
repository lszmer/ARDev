
using UnityEditor;
using UnityEngine;
using System.IO;

public class ModelQuantizerEditor
{
	// Updated to follow Sentis 2.1 docs: load ONNX via ModelLoader, quantize, then serialize
	[MenuItem("Sentis/Quantize YOLO Model to Uint8")]
	public static void QuantizeYoloModelToUint8()
	{
		// Project-relative asset path (for AssetDatabase) and absolute file path for existence check
		string relativeOnnxPath = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolo12n.onnx";
		string absoluteOnnxPath = Path.Combine(Application.dataPath, "PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/yolo12n.onnx");

		// Output filename and destination INSIDE Assets so Unity imports it as a ModelAsset
		string outputFileName = "yolo12n-uint8.sentis";
		string assetsRelativeOutputPath = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/" + outputFileName;
		string assetsAbsoluteOutputPath = Path.Combine(Application.dataPath, "PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/" + outputFileName);

		if (!File.Exists(absoluteOnnxPath))
		{
			Debug.LogError($"ONNX model not found at: {absoluteOnnxPath} (from {relativeOnnxPath})");
			return;
		}

		// Ensure the output directory exists under Assets
		var assetsOutputDir = Path.GetDirectoryName(assetsAbsoluteOutputPath);
		if (!Directory.Exists(assetsOutputDir))
		{
			Directory.CreateDirectory(assetsOutputDir);
		}

		try
		{
			Debug.Log($"Loading ONNX ModelAsset via Sentis: {relativeOnnxPath}");
			var modelAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(relativeOnnxPath);
			if (modelAsset == null)
			{
				Debug.LogError($"Failed to load ModelAsset from: {relativeOnnxPath}. Ensure the ONNX is imported by Sentis.");
				return;
			}
			Unity.InferenceEngine.Model model = null;
			try
			{
				model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
			}
			catch (System.Exception inner)
			{
				Debug.LogError($"ModelLoader failed for ModelAsset at: {relativeOnnxPath}. Exception: {inner.Message}");
				return;
			}
			if (model == null)
			{
				Debug.LogError("ModelLoader.Load(ModelAsset) returned null. Aborting quantization.");
				return;
			}

			Debug.Log("Quantizing model weights to Uint8. This may take a moment...");
			Unity.InferenceEngine.ModelQuantizer.QuantizeWeights(Unity.InferenceEngine.QuantizationType.Uint8, ref model);

			// Save inside Assets and trigger import so it becomes a Sentis ModelAsset
			Unity.InferenceEngine.ModelWriter.Save(assetsAbsoluteOutputPath, model);
			AssetDatabase.ImportAsset(assetsRelativeOutputPath);
			AssetDatabase.Refresh();

			Debug.Log($"<color=green>SUCCESS!</color> Quantized model saved and imported as ModelAsset:\n<b>{assetsRelativeOutputPath}</b>");
		}
		catch (System.Exception e)
		{
			Debug.LogError($"Failed to quantize model. Error: {e.Message}\n{e.StackTrace}");
		}
	}
}