using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MouseAgent : Agent
{
    [Header("Agent Stats")]
    public float moveSpeed = 5f; 
    
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
                // We start at 0.5f so it doesn't spawn literally inside the shelter
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

    // THE HIPPOCAMPUS: Feeding data to the Brain
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(rb.linearVelocity.magnitude);

        if (shelter != null)
        {
            Vector3 directionToShelter = (shelter.position - transform.position).normalized;
            float distanceToShelter = Vector3.Distance(shelter.position, transform.position);

            sensor.AddObservation(directionToShelter); 
            sensor.AddObservation(distanceToShelter);  
        }
    }

    // THE MOTOR CORTEX: Receiving commands from the Brain
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float moveX = actionBuffers.ContinuousActions[0];
        float moveZ = actionBuffers.ContinuousActions[1];

        Vector3 move = new Vector3(moveX, 0, moveZ);
        rb.linearVelocity = new Vector3(move.x * moveSpeed, rb.linearVelocity.y, move.z * moveSpeed);

        // Energy penalty 
        AddReward(-0.001f * rb.linearVelocity.magnitude);
    }

    // THE TRIGGERS: Death, False Alarms, Escapes, and Food
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Hawk"))
        {
            AddReward(-1.0f); // Eaten
            EndEpisode();
        }
        else if (other.CompareTag("Shelter"))
        {
            if (theHawk.isDiving == true)
            {
                AddReward(1.0f);  // TRUE ESCAPE!
                EndEpisode();
            }
            else
            {
                AddReward(-0.5f); // FALSE ALARM! 
                EndEpisode();
            }
        }
        else if (other.CompareTag("Food"))
        {
            AddReward(0.05f); // Positive reinforcement for exploring
            
            // Instantly teleport the food somewhere else so the mouse has to keep hunting
            other.transform.localPosition = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
        }
    }

    private void FixedUpdate()
    {
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
                
                // W4 Reaction Frames Penalty: e.g., -0.01f per frame of hesitation
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