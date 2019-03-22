using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SimpleDecision : MonoBehaviour, Decision
{
    public float learningRate;
    public float[] values = new float[4];
    private int action;
    private int lastAction;
    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        lastAction = action-1;
        if (++action > 4) action = 1;
        if (lastAction > -1)
        {
            values[lastAction] = values[lastAction] + learningRate * (reward - values[lastAction]);
        }
        return new float[] { action };
    }

    public List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}
