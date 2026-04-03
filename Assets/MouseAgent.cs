using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MouseAgent : Agent
{
    public float moveSpeed = 5f; 
    public HawkBot theHawk; 
    public Transform shelter; // ADDED: The link to the Safe Zone

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    // THE RESET: Randomize the map
    public override void OnEpisodeBegin()
    {
        // 1. Randomize Mouse
        transform.localPosition = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // 2. Randomize Shelter
        if (shelter != null)
        {
            shelter.localPosition = new Vector3(Random.Range(-8f, 8f), 0.5f, Random.Range(-8f, 8f));
        }

        // 3. Reset Hawk
        if (theHawk != null)
        {
            theHawk.ResetHawk();
        }
    }

    // THE HIPPOCAMPUS: Feeding data to the Brain
    public override void CollectObservations(VectorSensor sensor)
    {
        // Observation 1: Proprioception (How fast am I running?) -> 1 variable
        sensor.AddObservation(rb.linearVelocity.magnitude);

        // Observations 2 & 3: Place Cells (Where is home?) -> 4 variables
        if (shelter != null)
        {
            Vector3 directionToShelter = (shelter.position - transform.position).normalized;
            float distanceToShelter = Vector3.Distance(shelter.position, transform.position);

            sensor.AddObservation(directionToShelter); // X, Y, Z directions
            sensor.AddObservation(distanceToShelter);  // Distance
        }
    }

    // THE MOTOR CORTEX: Receiving commands from the Brain
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float moveX = actionBuffers.ContinuousActions[0];
        float moveZ = actionBuffers.ContinuousActions[1];

        Vector3 move = new Vector3(moveX, 0, moveZ);
        rb.linearVelocity = new Vector3(move.x * moveSpeed, rb.linearVelocity.y, move.z * moveSpeed);

        // Energy penalty (loss function part 1)
        AddReward(-0.001f * rb.linearVelocity.magnitude);
    }

    // THE TRIGGERS: Death, False Alarms, and True Escapes
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
                AddReward(1.0f);  // TRUE ESCAPE! Biological success.
                EndEpisode();
            }
            else
            {
                AddReward(-0.5f); // FALSE ALARM! Penalize cowardice.
                EndEpisode();
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