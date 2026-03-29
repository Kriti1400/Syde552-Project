using UnityEngine;

public class SmoothFollowCamera : MonoBehaviour
{
    [Header("Target Setup")]
    [Tooltip("Drag your Mouse GameObject here")]
    public Transform target; 

    [Header("Camera Dynamics")]
    [Tooltip("How far above and behind the target the camera sits")]
    public Vector3 offset = new Vector3(0f, 4f, -6f); 
    
    [Tooltip("Lower numbers = floatier camera. Higher numbers = snappier camera.")]
    public float smoothSpeed = 5f; 

    void LateUpdate()
    {
        // Safety check: if the target is destroyed or missing, don't crash
        if (target == null) return;

        // 1. Calculate exactly where the camera should hover relative to the mouse
        Vector3 desiredPosition = target.position + offset;

        // 2. Smoothly glide from the current position to the desired position
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);
        transform.position = smoothedPosition;

        // 3. Force the camera lens to always point directly at the center of the mouse
        transform.LookAt(target);
    }
}