
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Diagnostics;

public class ModelQuantizerEditor
{
	// Step-by-step, debuggable quantization flow with file pickers and timings.
	[MenuItem("Sentis/Tools/Quantize Model (Pick File)…")]
	public static void QuantizeModelInteractive()
	{
		// 1) Pick an input file under Assets (ONNX or Sentis ModelAsset)
		var inputPath = EditorUtility.OpenFilePanel("Select ONNX or Sentis Model", Application.dataPath, "onnx,sentis");
		if (string.IsNullOrEmpty(inputPath))
		{
			UnityEngine.Debug.LogWarning("Quantization cancelled: no input selected.");
			return;
		}
		if (!inputPath.StartsWith(Application.dataPath))
		{
			UnityEngine.Debug.LogError("Please select a file inside the project's Assets folder so Unity can import it.");
			return;
		}
		var relInputPath = "Assets" + inputPath.Substring(Application.dataPath.Length);

		// 2) Choose output file (Sentis .sentis) inside Assets
		var defaultOutName = Path.GetFileNameWithoutExtension(inputPath) + "-quant.sentis";
		var savePath = EditorUtility.SaveFilePanel("Save Quantized Sentis Model As", Path.GetDirectoryName(inputPath), defaultOutName, "sentis");
		if (string.IsNullOrEmpty(savePath))
		{
			UnityEngine.Debug.LogWarning("Quantization cancelled: no output selected.");
			return;
		}
		if (!savePath.StartsWith(Application.dataPath))
		{
			UnityEngine.Debug.LogError("Please save the output inside the project's Assets folder so Unity can import it.");
			return;
		}
		var relOutPath = "Assets" + savePath.Substring(Application.dataPath.Length);

		// 3) Ask quantization type
		var qTypes = new[] { "Uint8", "Float16", "None (no quantization)" };
		var picked = EditorUtility.DisplayDialogComplex("Quantization Type", "Choose quantization type:", qTypes[0], qTypes[1], qTypes[2]);
		Unity.InferenceEngine.QuantizationType? qType = null;
		switch (picked)
		{
			case 0: qType = Unity.InferenceEngine.QuantizationType.Uint8; break;
			case 1: qType = Unity.InferenceEngine.QuantizationType.Float16; break;
			case 2: qType = null; break;
			default: qType = Unity.InferenceEngine.QuantizationType.Uint8; break;
		}

		// 4) Load asset (handles .onnx or already .sentis ModelAsset)
		var swTotal = Stopwatch.StartNew();
		UnityEngine.Debug.Log($"[Sentis] Loading ModelAsset: {relInputPath}");
		var modelAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(relInputPath);
		if (modelAsset == null)
		{
			UnityEngine.Debug.LogError($"Could not load ModelAsset from {relInputPath}. Ensure the file is imported by Sentis.");
			return;
		}

		Unity.InferenceEngine.Model model = null;
		try
		{
			var sw = Stopwatch.StartNew();
			model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
			sw.Stop();
			UnityEngine.Debug.Log($"[Sentis] ModelLoader.Load() OK in {sw.ElapsedMilliseconds} ms");
		}
		catch (System.Exception e)
		{
			UnityEngine.Debug.LogError($"[Sentis] ModelLoader.Load() failed: {e.Message}\n{e.StackTrace}");
			return;
		}
		if (model == null)
		{
			UnityEngine.Debug.LogError("[Sentis] Model is null after loading.");
			return;
		}

		// 5) Optional quantization
		if (qType.HasValue)
		{
			try
			{
				var sw = Stopwatch.StartNew();
				UnityEngine.Debug.Log($"[Sentis] Quantizing weights to {qType.Value} …");
				Unity.InferenceEngine.ModelQuantizer.QuantizeWeights(qType.Value, ref model);
				sw.Stop();
				UnityEngine.Debug.Log($"[Sentis] Quantization finished in {sw.ElapsedMilliseconds} ms");
			}
			catch (System.Exception e)
			{
				UnityEngine.Debug.LogError($"[Sentis] Quantization failed: {e.Message}\n{e.StackTrace}");
				return;
			}
		}
		else
		{
			UnityEngine.Debug.Log("[Sentis] Skipping quantization (None selected)");
		}

		// 6) Save and import
		try
		{
			var sw = Stopwatch.StartNew();
			UnityEngine.Debug.Log($"[Sentis] Writing Sentis model: {relOutPath}");
			Unity.InferenceEngine.ModelWriter.Save(savePath, model);
			AssetDatabase.ImportAsset(relOutPath);
			AssetDatabase.Refresh();
			sw.Stop();
			UnityEngine.Debug.Log($"[Sentis] Save+Import finished in {sw.ElapsedMilliseconds} ms");
		}
		catch (System.Exception e)
		{
			UnityEngine.Debug.LogError($"[Sentis] Saving/importing failed: {e.Message}\n{e.StackTrace}");
			return;
		}

		swTotal.Stop();
		UnityEngine.Debug.Log($"<color=green>[Sentis]</color> Done. Total time: {swTotal.ElapsedMilliseconds} ms\nOutput: {relOutPath}");
	}

	// Convenience: End-to-end YOLO ONNX -> Sentis (+ NMS) -> optional quantize.
	[MenuItem("Sentis/Tools/Convert YOLO ONNX → Sentis (+NMS) …")]
	public static void ConvertYoloOnnxWithNms()
	{
		var onnxPath = EditorUtility.OpenFilePanel("Select YOLO ONNX (inside Assets)", Application.dataPath, "onnx");
		if (string.IsNullOrEmpty(onnxPath) || !onnxPath.StartsWith(Application.dataPath))
		{
			UnityEngine.Debug.LogError("Please select an ONNX file inside the project's Assets folder.");
			return;
		}
		var relOnnxPath = "Assets" + onnxPath.Substring(Application.dataPath.Length);

		// Thresholds
		var iouStr = EditorUtility.DisplayDialogComplex("IoU Threshold", "Choose IoU:", "0.6 (default)", "0.5", "Custom");
		float iou = iouStr == 1 ? 0.5f : 0.6f;
		if (iouStr == 2)
		{
			var custom = EditorUtility.DisplayDialog("Custom IoU", "Using default 0.6. Change in code if needed.", "OK");
		}
		// Score threshold selection with more options (0.2 - 0.6) and default 0.35
		float score = 0.35f;
		var scoreDlg = EditorUtility.DisplayDialogComplex("Score Threshold", "Choose Score:", "0.35 (default)", "0.2", "More…");
		if (scoreDlg == 0) { score = 0.35f; }
		else if (scoreDlg == 1) { score = 0.2f; }
		else
		{
			var moreDlg = EditorUtility.DisplayDialogComplex("Score Threshold (More)", "Choose Score:", "0.3", "0.5", "0.6");
			if (moreDlg == 0) score = 0.3f;
			else if (moreDlg == 1) score = 0.5f;
			else score = 0.6f;
		}

		var outPath = EditorUtility.SaveFilePanel("Save Sentis Model As", Path.GetDirectoryName(onnxPath), Path.GetFileNameWithoutExtension(onnxPath) + "-nms.sentis", "sentis");
		if (string.IsNullOrEmpty(outPath) || !outPath.StartsWith(Application.dataPath))
		{
			UnityEngine.Debug.LogError("Please save inside the project's Assets folder.");
			return;
		}
		var relOut = "Assets" + outPath.Substring(Application.dataPath.Length);

		// Optional quantization choice
		var q = EditorUtility.DisplayDialog("Quantize Weights?", "Quantize to Uint8 after NMS?", "Yes (Uint8)", "No");
		bool doQuant = q;

		try
		{
			var swTotal = Stopwatch.StartNew();
			UnityEngine.Debug.Log($"[Sentis] Loading ONNX as ModelAsset: {relOnnxPath}");
			var onnxAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(relOnnxPath);
			if (onnxAsset == null)
			{
				UnityEngine.Debug.LogError("Could not import ONNX as ModelAsset. Ensure Sentis importer is enabled for .onnx.");
				return;
			}

			var model = Unity.InferenceEngine.ModelLoader.Load(onnxAsset);
			if (model == null)
			{
				UnityEngine.Debug.LogError("ModelLoader.Load returned null for ONNX.");
				return;
			}

			// Build small graph to apply NMS, mirroring SentisModelEditorConverter but with logging
			var graph = new Unity.InferenceEngine.FunctionalGraph();
			var input = graph.AddInput(model, 0);
			var centersToCornersData = new[] { 1,0,1,0, 0,1,0,1, -0.5f,0,0.5f,0, 0,-0.5f,0,0.5f };
			var centersToCorners = Unity.InferenceEngine.Functional.Constant(new Unity.InferenceEngine.TensorShape(4, 4), centersToCornersData);
			var modelOutput = Unity.InferenceEngine.Functional.Forward(model, input)[0];
			var boxCoords = Unity.InferenceEngine.Functional.Transpose(modelOutput[0, ..4, ..], 0, 1);
			var allScores = Unity.InferenceEngine.Functional.Transpose(modelOutput[0, 4.., ..], 0, 1);
			var scores = Unity.InferenceEngine.Functional.ReduceMax(allScores, 1);
			var classIDs = Unity.InferenceEngine.Functional.ArgMax(allScores, 1);
			var boxCorners = Unity.InferenceEngine.Functional.MatMul(boxCoords, centersToCorners);
			var indices = Unity.InferenceEngine.Functional.NMS(boxCorners, scores, iou, score);
			var indices2 = Unity.InferenceEngine.Functional.BroadcastTo(Unity.InferenceEngine.Functional.Unsqueeze(indices, -1), new[] { 4 });
			var labelIDs = Unity.InferenceEngine.Functional.Gather(classIDs, 0, indices);
			var coords = Unity.InferenceEngine.Functional.Gather(boxCoords, 0, indices2);
			var modelFinal = graph.Compile(coords, labelIDs);

			if (doQuant)
			{
				UnityEngine.Debug.Log("[Sentis] Quantizing final graph to Uint8 …");
				Unity.InferenceEngine.ModelQuantizer.QuantizeWeights(Unity.InferenceEngine.QuantizationType.Uint8, ref modelFinal);
			}

			UnityEngine.Debug.Log($"[Sentis] Saving Sentis model: {relOut}");
			Unity.InferenceEngine.ModelWriter.Save(outPath, modelFinal);
			AssetDatabase.ImportAsset(relOut);
			AssetDatabase.Refresh();
			swTotal.Stop();
			UnityEngine.Debug.Log($"<color=green>[Sentis]</color> YOLO ONNX → Sentis (+NMS) {(doQuant ? "+ Uint8" : "")} OK in {swTotal.ElapsedMilliseconds} ms\nOutput: {relOut}");
		}
		catch (System.Exception e)
		{
			UnityEngine.Debug.LogError($"[Sentis] Conversion failed: {e.Message}\n{e.StackTrace}");
		}
	}
}