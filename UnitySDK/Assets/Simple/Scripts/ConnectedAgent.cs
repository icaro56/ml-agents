using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using MLAgents;


public class ConnectedAgent : Agent {
    
    public BanditCube currentBandit;
    public Material agentMaterial;
    private Material resetMaterial;
    private BanditCube nextBandit;

    private ConnectedAcademy connectedAcademy;

    public void Start()
    {
        connectedAcademy = Object.FindObjectOfType<ConnectedAcademy>() as ConnectedAcademy;
    }
    
    public override void CollectObservations()
    {        
        currentBandit = nextBandit;
        if (!currentBandit && connectedAcademy)
        {
            currentBandit = connectedAcademy.GetNewBandit();            
        }
        
        AddVectorObs(currentBandit.index);
    }    
        
    public override void AgentAction(float[] vectorAction, string textAction)
	{
        Thread.Sleep(1000);
        int action = (int)vectorAction[0];
        if (currentBandit)
        {            
            nextBandit = currentBandit.ChoosePath(action);
            if (nextBandit != null)
            {
                nextBandit.SetMaterial(agentMaterial);
                currentBandit.ResetMaterial();
                AddReward(currentBandit.reward);
                if (currentBandit.reward == 1.0f)
                {
                    Done();
                }
            }   
        }        
    }

    public override void AgentReset()
    {
        if(currentBandit)
            currentBandit.ResetMaterial();

        currentBandit = null;
        nextBandit = connectedAcademy.GetNewBandit();
        nextBandit.SetMaterial(agentMaterial);
    }

    public override void AgentOnDone()
    {

    }

    /*public Academy academy;
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
    }*/
}
