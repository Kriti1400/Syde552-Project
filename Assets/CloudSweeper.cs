using UnityEngine;

public class CloudSweeper : MonoBehaviour
{
    [Header("Movement Settings")]
    public float speed = 5f;
    public Vector3 moveDirection = Vector3.right; // Moves along the X axis by default

    [Header("Loop Boundaries")]
    public float despawnX = 25f;  // How far it goes before teleporting
    public float respawnX = -25f; // Where it teleports back to

    [Header("Spawn Settings")]
    public float spawnRadiusXZ = 8f; // Within 8 from the middle
    public float minSpawnY = 30f;
    public float maxSpawnY = 40f;

    void Start()
    {
        // Pick a random starting position the moment the game hits Play
        float startX = Random.Range(-spawnRadiusXZ, spawnRadiusXZ);
        float startY = Random.Range(minSpawnY, maxSpawnY);
        float startZ = Random.Range(-spawnRadiusXZ, spawnRadiusXZ);

        transform.position = new Vector3(startX, startY, startZ);
    }

    void Update()
    {
        // Move the cloud smoothly every frame
        transform.Translate(moveDirection.normalized * speed * Time.deltaTime, Space.World);

        // Check if it has crossed the despawn boundary
        if (transform.position.x > despawnX)
        {
            // Teleport back to the start, but pick a NEW random height and depth!
            float newY = Random.Range(minSpawnY, maxSpawnY);
            float newZ = Random.Range(-spawnRadiusXZ, spawnRadiusXZ);
            
            transform.position = new Vector3(respawnX, newY, newZ);
        }
    }
}