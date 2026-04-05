using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MouseAgent : Agent
{
    [Header("Agent Stats")]
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
        if (rb == null) Debug.LogError("STOP! Attach a Rigidbody to the Mouse object!");
    }

    public override void OnEpisodeBegin()
    {
        if (shelter == null || theHawk == null) return;

        // 1. Randomize Shelter
        shelter.localPosition = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
        
        // 2. DYNAMIC STRATIFIED SPAWNING
        float timeToImpact = theHawk.cruiseAltitude / theHawk.diveSpeed;
        float survivalRadius = moveSpeed * timeToImpact;
        float maxForageRadius = survivalRadius * 2f;
        
        float spawnDistance = (Random.value > 0.5f) ? Random.Range(0.5f, survivalRadius) : Random.Range(survivalRadius, maxForageRadius);
        Vector2 randomDir = Random.insideUnitCircle.normalized; 
        Vector3 spawnPos = shelter.localPosition + new Vector3(randomDir.x * spawnDistance, 0f, randomDir.y * spawnDistance);
        
        // BOUNDARY CLAMP: Prevent spawning off-plane
        spawnPos.x = Mathf.Clamp(spawnPos.x, -8.5f, 8.5f);
        spawnPos.z = Mathf.Clamp(spawnPos.z, -8.5f, 8.5f);
        transform.localPosition = new Vector3(spawnPos.x, 0.5f, spawnPos.z);

        threatDetected = false;
        hasReacted = false;
        framesElapsed = 0;

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        theHawk.ResetHawk();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (rb == null) return;
        sensor.AddObservation(rb.linearVelocity.magnitude);

        if (shelter != null)
        {
            sensor.AddObservation((shelter.position - transform.position).normalized); 
            sensor.AddObservation(Vector3.Distance(shelter.position, transform.position));  
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (rb == null) return;
        float moveX = actionBuffers.ContinuousActions[0];
        float moveZ = actionBuffers.ContinuousActions[1];

        rb.linearVelocity = new Vector3(moveX * moveSpeed, rb.linearVelocity.y, moveZ * moveSpeed);

        // W2 Energy penalty 
        AddReward(-0.001f * rb.linearVelocity.magnitude);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Hawk")) { AddReward(-1.0f); EndEpisode(); }
        else if (other.CompareTag("Shelter"))
        {
            if (theHawk.isDiving) { AddReward(1.0f); EndEpisode(); }
            else { AddReward(-0.5f); EndEpisode(); } // W3 False Alarm 
        }
    }

    private void FixedUpdate()
    {
        // VOID PENALTY: Prevents cheating by falling off the map
        if (transform.localPosition.y < -0.5f) { AddReward(-1.0f); EndEpisode(); return; }

        // W5 CONTINUOUS FORAGING REWARD: Rewards being outside shelter 
        if (shelter != null && Vector3.Distance(transform.position, shelter.position) > 1.5f)
        {
            AddReward(0.0005f); 
        }

        // W4 REACTION LOGIC: Enforces subsecond reflexes 
        if (theHawk != null && theHawk.isDiving && !threatDetected) { threatDetected = true; }
        if (threatDetected && !hasReacted)
        {
            framesElapsed++;
            if (rb != null && rb.linearVelocity.magnitude > 0.1f) 
            {
                hasReacted = true;
                AddReward(-0.01f * framesElapsed); 
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;
        cont[0] = Input.GetAxisRaw("Horizontal");
        cont[1] = Input.GetAxisRaw("Vertical");
    }
}