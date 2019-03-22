using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SimpleAgent : Agent {

    public Bandit currentBandit;
    public Bandit[] bandits;
    public override void CollectObservations()
    {
        var bandit = Random.Range(0, bandits.Length);
        currentBandit = bandits[bandit];
        AddVectorObs(bandit);
    }    
        
    public override void AgentAction(float[] vectorAction, string textAction)
	{
        int action = (int)vectorAction[0];
        if(currentBandit) AddReward(currentBandit.PullArm(action));
    }

    public override void AgentReset()
    {
        if (currentBandit) currentBandit.Reset();
    }

    public override void AgentOnDone()
    {

    }

    public Academy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;
    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    private void WaitTimeInference()
    {
        if (!academy.GetIsInference())
        {
            RequestDecision();
        }
        else
        {
            if (timeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                timeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                timeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }
}
