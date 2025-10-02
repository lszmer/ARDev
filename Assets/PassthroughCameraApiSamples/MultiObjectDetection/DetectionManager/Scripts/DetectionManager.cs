// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using System.Collections.Generic;
using Meta.XR.Samples;
using UnityEngine;
using UnityEngine.Events;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class DetectionManager : MonoBehaviour
    {
        [SerializeField] private WebCamTextureManager m_webCamTextureManager;

        [Header("Controls configuration")]
		[SerializeField] private OVRInput.RawButton m_actionButton = OVRInput.RawButton.A;
		[SerializeField] private OVRInput.RawButton m_pointerButton = OVRInput.RawButton.RHandTrigger;
		[SerializeField] private OVRInput.RawButton m_deleteAllButton = OVRInput.RawButton.B;

        [Header("Ui references")]
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;

        [Header("Placement configureation")]
        [SerializeField] private GameObject m_spwanMarker;
        [SerializeField] private EnvironmentRayCastSampleManager m_environmentRaycast;
        [SerializeField] private float m_spawnDistance = 0.25f;
        [SerializeField] private AudioSource m_placeSound;
		[SerializeField] private Transform m_rightHandAnchor;
		[SerializeField] private float m_pointerMaxDistance = 3.0f;
		[SerializeField] private float m_pointerSelectionThreshold = 0.12f;

        [Header("Sentis inference ref")]
        [SerializeField] private SentisInferenceRunManager m_runInference;
        [SerializeField] private SentisInferenceUiManager m_uiInference;
        [Space(10)]
        public UnityEvent<int> OnObjectsIdentified;

        private bool m_isPaused = true;
        private List<GameObject> m_spwanedEntities = new();
        private bool m_isStarted = false;
        private bool m_isSentisReady = false;
        private float m_delayPauseBackTime = 0;
		private LineRenderer m_pointerLine;
		private int m_selectedBoxIndex = -1;

        #region Unity Functions
		private void Awake()
		{
			OVRManager.display.RecenteredPose += CleanMarkersCallBack;
			// Ensure pause state toggles when the UI menu starts/stops
			if (m_uiMenuManager)
			{
				m_uiMenuManager.OnPause.AddListener(OnPause);
			}
		}

		private IEnumerator Start()
        {
            // Wait until Sentis model is loaded
            var sentisInference = FindAnyObjectByType<SentisInferenceRunManager>();
            while (!sentisInference.IsModelLoaded)
            {
                yield return null;
            }
			m_isSentisReady = true;

			// Create a simple pointer line for right-hand aiming
			if (!m_pointerLine)
			{
				var go = new GameObject("RightHandPointer");
				m_pointerLine = go.AddComponent<LineRenderer>();
				m_pointerLine.positionCount = 2;
				m_pointerLine.startWidth = 0.003f;
				m_pointerLine.endWidth = 0.0015f;
				m_pointerLine.material = new Material(Shader.Find("Sprites/Default"));
				m_pointerLine.startColor = new Color(0.1f, 0.7f, 1.0f, 0.9f);
				m_pointerLine.endColor = new Color(0.1f, 0.7f, 1.0f, 0.3f);
				m_pointerLine.enabled = false;
			}
        }

		private void Update()
        {
            // Get the WebCamTexture CPU image
            var hasWebCamTextureData = m_webCamTextureManager.WebCamTexture != null;

            if (!m_isStarted)
            {
                // Manage the Initial Ui Menu
                if (hasWebCamTextureData && m_isSentisReady)
                {
                    m_uiMenuManager.OnInitialMenu(m_environmentRaycast.HasScenePermission());
                    m_isStarted = true;
                }
            }
			else
            {
				// Pointer handling while holding the right side button
				var isPointerActive = OVRInput.Get(m_pointerButton);
				HandlePointer(isPointerActive);

				// Press A button: place markers for all current detections (simplified troubleshooting path)
				if (OVRInput.GetUp(m_actionButton) && m_delayPauseBackTime <= 0)
				{
					Debug.Log("DetectionManager: A pressed -> place all current detections");
					m_placeSound?.Play();
					SpwanCurrentDetectedObjects();
				}

				// Press B button to delete all markers
				if (OVRInput.GetUp(m_deleteAllButton))
				{
					DeleteAllMarkers();
				}
                // Cooldown for the A button after return from the pause menu
                m_delayPauseBackTime -= Time.deltaTime;
                if (m_delayPauseBackTime <= 0)
                {
                    m_delayPauseBackTime = 0;
                }
            }

            // Not start a sentis inference if the app is paused or we don't have a valid WebCamTexture
			if (m_isPaused || !hasWebCamTextureData)
            {
				if (m_isPaused)
                {
                    // Set the delay time for the A button to return from the pause menu
                    m_delayPauseBackTime = 0.1f;
					// Debug to verify pause gating
					// Debug.Log("DetectionManager: Update gated by pause");
                }
                return;
            }

            // Run a new inference when the current inference finishes
            if (!m_runInference.IsRunning())
            {
                m_runInference.RunInference(m_webCamTextureManager.WebCamTexture);
            }
        }
        #endregion

		#region Marker Functions
		private void DeleteAllMarkers()
		{
			foreach (var e in m_spwanedEntities)
			{
				if (e)
				{
					Destroy(e);
				}
			}
			m_spwanedEntities.Clear();
			OnObjectsIdentified?.Invoke(0);
		}
        /// <summary>
        /// Clean 3d markers when the tracking space is re-centered.
        /// </summary>
        private void CleanMarkersCallBack()
        {
            foreach (var e in m_spwanedEntities)
            {
                Destroy(e, 0.1f);
            }
            m_spwanedEntities.Clear();
            OnObjectsIdentified?.Invoke(-1);
        }

		private void HandlePointer(bool active)
		{
			if (!m_rightHandAnchor || m_isPaused)
			{
				if (m_pointerLine) m_pointerLine.enabled = false;
				m_selectedBoxIndex = -1;
				return;
			}

			if (!active)
			{
				if (m_pointerLine) m_pointerLine.enabled = false;
				m_selectedBoxIndex = -1;
				return;
			}

			var ray = new Ray(m_rightHandAnchor.position, m_rightHandAnchor.forward);
			UpdatePointerVisual(ray);
			m_selectedBoxIndex = FindClosestBoxToRay(ray, m_pointerMaxDistance, m_pointerSelectionThreshold);
		}

		private void UpdatePointerVisual(Ray ray)
		{
			if (!m_pointerLine) return;
			m_pointerLine.enabled = true;
			var start = ray.origin;
			var end = ray.origin + ray.direction * m_pointerMaxDistance;
			m_pointerLine.SetPosition(0, start);
			m_pointerLine.SetPosition(1, end);
		}

		private int FindClosestBoxToRay(Ray ray, float maxDistance, float maxPerpendicularDist)
		{
			var bestIndex = -1;
			var bestMetric = float.MaxValue;
			for (var i = 0; i < m_uiInference.BoxDrawn.Count; i++)
			{
				var box = m_uiInference.BoxDrawn[i];
				if (!box.WorldPos.HasValue) continue;
				var p = box.WorldPos.Value;
				var toP = p - ray.origin;
				var along = Vector3.Dot(toP, ray.direction);
				if (along < 0 || along > maxDistance) continue;
				var perpendicular = Vector3.Cross(ray.direction, toP).magnitude;
				if (perpendicular > maxPerpendicularDist) continue;
				// Prefer closer along-ray distance with small perpendicular
				var metric = perpendicular * 10.0f + Mathf.Abs(along);
				if (metric < bestMetric)
				{
					bestMetric = metric;
					bestIndex = i;
				}
			}
			return bestIndex;
		}
        /// <summary>
        /// Spwan 3d markers for the detected objects
        /// </summary>
		private void SpwanCurrentDetectedObjects()
        {
            var count = 0;
            foreach (var box in m_uiInference.BoxDrawn)
            {
				// Prefer placing at the world-space position where the box UI is drawn
				var placePos = box.WorldPos.HasValue ? box.WorldPos : box.WorldUiPos;
				if (PlaceMarkerUsingEnvironmentRaycast(placePos, box.ClassName))
                {
                    count++;
                }
            }
            if (count > 0)
            {
				// Play sound if a new marker is placed.
				m_placeSound?.Play();
            }
            OnObjectsIdentified?.Invoke(count);
        }

        /// <summary>
        /// Place a marker using the environment raycast
        /// </summary>
		private bool PlaceMarkerUsingEnvironmentRaycast(Vector3? position, string className)
        {
            // Check if the position is valid
            if (!position.HasValue)
            {
                return false;
            }

            // Check if you spanwed the same object before
            var existMarker = false;
            foreach (var e in m_spwanedEntities)
            {
                var markerClass = e.GetComponent<DetectionSpawnMarkerAnim>();
                if (markerClass)
                {
                    var dist = Vector3.Distance(e.transform.position, position.Value);
                    if (dist < m_spawnDistance && markerClass.GetYoloClassName() == className)
                    {
                        existMarker = true;
                        break;
                    }
                }
            }

			if (!existMarker)
            {
                // spawn a visual marker
                var eMarker = Instantiate(m_spwanMarker);
                m_spwanedEntities.Add(eMarker);

                // Update marker transform with the real world transform
                eMarker.transform.SetPositionAndRotation(position.Value, Quaternion.identity);
                eMarker.GetComponent<DetectionSpawnMarkerAnim>().SetYoloClassName(className);
            }

            return !existMarker;
        }
        #endregion

        #region Public Functions
        /// <summary>
        /// Pause the detection logic when the pause menu is active
        /// </summary>
        public void OnPause(bool pause)
        {
            m_isPaused = pause;
        }
        #endregion
    }
}
