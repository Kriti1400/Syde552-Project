using UnityEngine;

public class HawkBot : MonoBehaviour
{
    [Header("Hunting Targets")]
    public Transform prey;          // Drag the Mouse here in the Inspector
    private Rigidbody preyRb;       // Added: To check the mouse's velocity for motion vision
    
    [Header("Flight Dynamics")]
    public float cruiseSpeed = 4f;  
    public float diveSpeed = 6f;    
    public float cruiseAltitude = 10f;
    
    [Header("Attack Timing")]
    public float minDiveWait = 3f;  
    public float maxDiveWait = 8f;  

    public bool isDiving = false;
    private float diveTimer = 0f;
    private float nextDiveTime;
    private Vector3 lockedDivePosition;

    void Start()
    {
        // Get the mouse's physics body so we can read its speed later
        if (prey != null) 
        {
            preyRb = prey.GetComponent<Rigidbody>();
        }
        
        nextDiveTime = Random.Range(minDiveWait, maxDiveWait);
    }

    void Update()
    {
        if (prey == null) return;

        if (!isDiving)
        {
            // STATE 1: CRUISING
            Vector3 cruisePosition = new Vector3(prey.position.x, cruiseAltitude, prey.position.z);
            transform.position = Vector3.MoveTowards(transform.position, cruisePosition, cruiseSpeed * Time.deltaTime);

            diveTimer += Time.deltaTime;
            if (diveTimer >= nextDiveTime)
            {
                // Trigger the attack!
                isDiving = true;
                diveTimer = 0f; 

                // EMERGENT FREEZING LOGIC (The Hawk's Motion Vision)
                // If the mouse's velocity is basically zero, it blends in!
                if (preyRb != null && preyRb.linearVelocity.magnitude < 0.1f)
                {
                    // 70% chance to miss the locked position
                    if (Random.value < 0.70f)
                    {
                        // Generate a random error offset (between 2 and 4 meters away)
                        Vector3 missOffset = new Vector3(Random.Range(-4f, 4f), 0, Random.Range(-4f, 4f));
                        lockedDivePosition = prey.position + missOffset;
                    }
                    else
                    {
                        // 30% chance the hawk still spots the frozen mouse
                        lockedDivePosition = prey.position;
                    }
                }
                else
                {
                    // The mouse is moving! Perfect visual lock.
                    lockedDivePosition = prey.position;
                }
            }
        }
        else
        {
            // STATE 2: THE DIVE
            
            // If the mouse is moving during the dive, the hawk tracks it
            // If the mouse is frozen, the hawk stays locked on its original (potentially missed) coordinate
            if (preyRb != null && preyRb.linearVelocity.magnitude > 0.1f)
            {
                lockedDivePosition = prey.position; // Continuously update target
            }

            transform.position = Vector3.MoveTowards(transform.position, lockedDivePosition, diveSpeed * Time.deltaTime);

            if (Vector3.Distance(transform.position, lockedDivePosition) < 0.5f)
            {
                // Reset
                isDiving = false;
                nextDiveTime = Random.Range(minDiveWait, maxDiveWait);
                transform.position = new Vector3(transform.position.x, cruiseAltitude, transform.position.z);
            }
        }
    }

    public void ResetHawk()
    {
        isDiving = false;
        diveTimer = 0f;
        nextDiveTime = Random.Range(minDiveWait, maxDiveWait);

        float randomX = Random.Range(-8f, 8f);
        float randomZ = Random.Range(-8f, 8f);
        transform.position = new Vector3(randomX, cruiseAltitude, randomZ);
    }
}