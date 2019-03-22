using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MLAgents;

public class ConnectedDecision : MonoBehaviour, Decision
{
    public float learningRate = .9f;
    public float gamma = .9f;
    public float explorationEpsilon;
    
    private float[][] q;    
    private int lastAction, lastState; 
    private int action;

    public float[][] Q
    {
        get
        {
            if (q ==  null)
            {
                var connectedAcademy = Object.FindObjectOfType<ConnectedAcademy>() as ConnectedAcademy;                
                q = new float[connectedAcademy.bandits.Length][];
                foreach(var bandit in connectedAcademy.bandits)
                {
                    if (bandit.Connections.Count == 0)
                    {
                        q = null;
                        return q;
                    }
                    q[bandit.index] = new float[bandit.Connections.Count];
                }                
            }
            return q;
        }
    }    

    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {

        if (Q == null)
            return new float[] { 0 };

        lastAction = action-1;        
        var state = (int)vectorObs[0];

        if (lastAction > -1)
        {            
            Q[lastState][lastAction] = Q[lastState][lastAction] + learningRate * (reward + gamma * Q[state].Max()) - Q[lastState][lastAction];
        }
        lastState = state;

        return DecideAction(Q[state].ToList());
    }

    public float[] DecideAction(List<float> state)
    {
        var r = Random.Range(0.0f, 1.0f);
        explorationEpsilon = Mathf.Min(explorationEpsilon - .0001f, .1f);
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
