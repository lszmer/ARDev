using UnityEngine;

namespace PassthroughCameraSamples.StartScene
{
	// Ensures any stray model picker or legacy menus are removed when entering the Start scene
	public class StartMenuCleanup : MonoBehaviour
	{
		private void Start()
		{
			var strayPicker = GameObject.Find("PreloadModelPicker");
			if (strayPicker)
			{
				Destroy(strayPicker);
			}
			var strayLegacy = GameObject.Find("ModelSelectionMenu");
			if (strayLegacy)
			{
				Destroy(strayLegacy);
			}
		}
	}
}


