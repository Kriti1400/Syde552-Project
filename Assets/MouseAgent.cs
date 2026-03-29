using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MouseAgent : Agent
{
    public float moveSpeed = 20f;
    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Randomly place the mouse at the start of each simulation
        transform.localPosition = new Vector3(Random.Range(-15f, 15f), 0.5f, Random.Range(-15f, 15f));
        rb.linearVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(rb.linearVelocity.magnitude);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Get continuous movement from the Neural Network
        float moveX = actionBuffers.ContinuousActions[0];
        float moveZ = actionBuffers.ContinuousActions[1];

        Vector3 move = new Vector3(moveX, 0, moveZ);
        rb.AddForce(move * moveSpeed);

        // Energy penalty (loss function part 1)
        AddReward(-0.001f * rb.linearVelocity.magnitude);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Hawk") || other.gameObject.CompareTag("Snake"))
        {
            AddReward(-1.0f); // Death penalty
            EndEpisode();
        }
        else if (other.gameObject.CompareTag("Shelter"))
        {
            AddReward(1.0f);  // Survival reward
            EndEpisode();
        }
    }

    // Allows you to test it yourself with WASD keys before we plug Python in
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}