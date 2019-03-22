using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using System.Linq;

public class ContextualDecision : MonoBehaviour, Decision
{
    public float learningRate;
    public float explorationEpsilon;
    public float[][] q;    
    private int lastAction, lastState; 
    private int action;

    public void Start()
    {
        q = new float[4][];
        for (int i = 0; i < 4; i++)
        {
            q[i] = new float[4];
        }
    }

    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        lastAction = action-1;
        //if (++action > 4) action = 1;
        if (lastAction > -1)
        {
            q[lastState][lastAction] = q[lastState][lastAction] + learningRate * (reward - q[lastState][lastAction]);
        }
        lastState = (int)vectorObs[0];
        return DecideAction(q[lastState].ToList());
    }

    public float[] DecideAction(List<float> state)
    {
        var r = Random.Range(0.0f, 1.0f);
        explorationEpsilon = Mathf.Max(explorationEpsilon - .0001f, .1f);
        if (r < explorationEpsilon)
        {
            action = RandomAction(state) + 1;
        }
        else
        {
            action = GetAction(state) + 1;
        }
        return new float[] { action };
    }

    private int RandomAction(List<float> states)
    {
        return Random.Range(0, states.Count);
    }

    private int GetAction(List<float> values)
    {
        float maxValue = values.Max();
        return values.ToList().IndexOf(maxValue);
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
