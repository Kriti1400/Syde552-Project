using UnityEngine;

public class HawkBot : MonoBehaviour
{
    [Header("Hunting Targets")]
    public Transform prey;          // Drag the Mouse here in the Inspector
    
    [Header("Flight Dynamics")]
    public float cruiseSpeed = 4f;  // Speed while hovering high up
    public float diveSpeed = 6f;    // Gives mouse ~1.4 seconds to react
    public float cruiseAltitude = 10f;
    
    [Header("Attack Timing")]
    public float minDiveWait = 3f;  // Minimum seconds before a dive
    public float maxDiveWait = 8f;  // Maximum seconds before a dive

    private bool isDiving = false;
    private float diveTimer = 0f;
    private float nextDiveTime;
    private Vector3 lockedDivePosition;

    void Start()
    {
        // Pick a random time for the very first attack
        nextDiveTime = Random.Range(minDiveWait, maxDiveWait);
    }

    void Update()
    {
        if (prey == null) return;

        if (!isDiving)
        {
            // STATE 1: CRUISING (Casting the shadow)
            Vector3 cruisePosition = new Vector3(prey.position.x, cruiseAltitude, prey.position.z);
            transform.position = Vector3.MoveTowards(transform.position, cruisePosition, cruiseSpeed * Time.deltaTime);

            // Tick the timer
            diveTimer += Time.deltaTime;
            if (diveTimer >= nextDiveTime)
            {
                // Trigger the attack!
                isDiving = true;
                lockedDivePosition = prey.position; 
                diveTimer = 0f; 
            }
        }
        else
        {
            // STATE 2: THE DIVE (The looming stimulus)
            transform.position = Vector3.MoveTowards(transform.position, lockedDivePosition, diveSpeed * Time.deltaTime);

            // Check if we hit the floor (Y roughly equals 0)
            if (Vector3.Distance(transform.position, lockedDivePosition) < 0.5f)
            {
                // Reset back to cruising state
                isDiving = false;
                nextDiveTime = Random.Range(minDiveWait, maxDiveWait);
                
                // Instantly snap back to altitude
                transform.position = new Vector3(transform.position.x, cruiseAltitude, transform.position.z);
            }
        }
    }
    public void ResetHawk()
    {
        isDiving = false;
        diveTimer = 0f;
        nextDiveTime = Random.Range(minDiveWait, maxDiveWait);

        // Teleport the hawk to a random location in the sky.
        // Adjust the -10f and 10f to match the size of your green floor!
        float randomX = Random.Range(-10f, 10f);
        float randomZ = Random.Range(-10f, 10f);
        
        transform.position = new Vector3(randomX, cruiseAltitude, randomZ);
    }
}