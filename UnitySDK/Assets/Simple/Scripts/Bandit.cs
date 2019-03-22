using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bandit : MonoBehaviour
{
    public SimpleArm[] arms;
    private MeshRenderer mesh;
    private Material reset;

    // Use this for initialization
    void Start()
    {
        mesh = GetComponent<MeshRenderer>();
        reset = mesh.material;
    }

    public int PullArm(int arm)
    {
        if (arm < 0 || arm > arms.Length) return 0;
        mesh.material = arms[arm - 1].material;
        return arms[arm - 1].rewardValue;
    }

    public void Reset()
    {
        mesh.material = reset;
    }
}