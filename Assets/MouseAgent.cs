using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MouseAgent : Agent
{
    [Header("Agent Stats")]
    // Biologically constrained to ~3.5m/s (max sprint speed of Mus musculus)
    public float moveSpeed = 3.5f; 
    
    [Header("Environment Links")]
    public HawkBot theHawk; 
    public Transform shelter; 

    private Rigidbody rb;
    private bool threatDetected = false;
    private bool hasReacted = false;
    private int framesElapsed = 0;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    // THE RESET: Randomize the map using dynamic physical thresholds
    public override void OnEpisodeBegin()
    {
        // 1. Randomize Shelter Location
        if (shelter != null)
        {
            shelter.localPosition = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
            
            // 2. DYNAMIC STRATIFIED SPAWNING
            // Calculate exactly how long the mouse has before the hawk hits the floor
            float timeToImpact = theHawk.cruiseAltitude / theHawk.diveSpeed;
            
            // Calculate the absolute maximum distance the mouse could run in that time
            float survivalRadius = moveSpeed * timeToImpact;
            float maxForageRadius = survivalRadius * 2f;
            
            float spawnDistance;
            
            // Flip a coin (50% chance)
            if (Random.value > 0.5f)
            {
                // Scenario A: Spawn inside the Flight Zone (Close enough to survive)
                spawnDistance = Random.Range(0.5f, survivalRadius);
            }
            else
            {
                // Scenario B: Spawn inside the Freeze Zone (Mathematically too far to run)
                spawnDistance = Random.Range(survivalRadius, maxForageRadius);
            }

            // Pick a random direction, then multiply by our calculated distance
            Vector2 randomDirection = Random.insideUnitCircle.normalized; 
            Vector3 spawnPos = shelter.localPosition + new Vector3(randomDirection.x * spawnDistance, 0f, randomDirection.y * spawnDistance);
            
            // --- BOUNDARY CLAMPING ---
            // Forces the spawn position to stay inside the 8x8 plane boundary
            spawnPos.x = Mathf.Clamp(spawnPos.x, -8.5f, 8.5f);
            spawnPos.z = Mathf.Clamp(spawnPos.z, -8.5f, 8.5f);
            
            // Keep the Y value at 0.5f so the mouse doesn't spawn under the floor
            transform.localPosition = new Vector3(spawnPos.x, 0.5f, spawnPos.z);

            threatDetected = false;
            hasReacted = false;
            framesElapsed = 0;
        }

        // Kill any sliding momentum from the last episode
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 3. Reset Hawk
        if (theHawk != null)
        {
            theHawk.ResetHawk();
        }
    }

    // THE HIPPOCAMPUS / RETROSPLENIAL CORTEX: Feeding spatial data to the Brain
    public override void CollectObservations(VectorSensor sensor)
    {
        // 1 observation
        sensor.AddObservation(rb.linearVelocity.magnitude);

        if (shelter != null)
        {
            Vector3 directionToShelter = (shelter.position - transform.position).normalized;
            float distanceToShelter = Vector3.Distance(shelter.position, transform.position);

            // 3 observations (X, Y, Z)
            sensor.AddObservation(directionToShelter); 
            // 1 observation
            sensor.AddObservation(distanceToShelter);  
        }
        // Total = 5 observations (Make sure Vector Space Size is set to 5 in the Inspector!)
    }

    // THE MOTOR CORTEX: Receiving commands from the Brain
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float moveX = actionBuffers.ContinuousActions[0];
        float moveZ = actionBuffers.ContinuousActions[1];

        Vector3 move = new Vector3(moveX, 0, moveZ);
        rb.linearVelocity = new Vector3(move.x * moveSpeed, rb.linearVelocity.y, move.z * moveSpeed);

        // W2 Energy penalty: Punish sprinting
        AddReward(-0.001f * rb.linearVelocity.magnitude);
    }

    // THE TRIGGERS: Death, False Alarms, and True Escapes
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Hawk"))
        {
            AddReward(-1.0f); // W1 Death Penalty: Eaten
            EndEpisode();
        }
        else if (other.CompareTag("Shelter"))
        {
            if (theHawk.isDiving == true)
            {
                AddReward(1.0f);  // TRUE ESCAPE! Reward the fast pathway
                EndEpisode();
            }
            else
            {
                AddReward(-0.5f); // W3 False Alarm: Punish running from harmless clouds
                EndEpisode();
            }
        }
    }

    private void FixedUpdate()
    {
        // --- THE VOID PENALTY ---
        // If the mouse falls off the plane, kill it instantly
        if (transform.localPosition.y < -1f)
        {
            AddReward(-1.0f);
            EndEpisode();
            return; 
        }

        // --- W5 FORAGING REWARD ---
        // A continuous trickle of points for grazing out in the open
        if (shelter != null && Vector3.Distance(transform.position, shelter.position) > 1.5f)
        {
            AddReward(0.0005f); 
        }

        // --- REFLEX LOGIC ---
        // 1. Check if the hawk just started diving
        if (theHawk != null && theHawk.isDiving && !threatDetected)
        {
            threatDetected = true;
        }

        // 2. If the threat is active, but the mouse hasn't moved yet, count the frames
        if (threatDetected && !hasReacted)
        {
            framesElapsed++;

            // 3. The moment the mouse outputs a velocity vector, calculate the penalty
            if (rb.linearVelocity.magnitude > 0.1f) 
            {
                hasReacted = true;
                
                // W4 Reaction Frames Penalty: Punish the retino-collicular pathway for hesitation
                AddReward(-0.01f * framesElapsed); 
            }
        }
    }

    // Allows you to test it manually with WASD keys
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxisRaw("Horizontal");
        continuousActionsOut[1] = Input.GetAxisRaw("Vertical");
    }
}