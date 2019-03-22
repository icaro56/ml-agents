using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class ConnectedAcademy : Academy {
    public GameObject BanditEnvironment;
    public BanditCube[] bandits;

    public override void AcademyReset()
    {
        if (BanditEnvironment)
        {
            bandits = BanditEnvironment.GetComponentsInChildren<BanditCube>();
            int i = 0;
            foreach (var bandit in bandits)
            {
                bandit.index = i++;
            }
            Debug.Log("Creating children");
        }
        else
        {
            throw new UnityAgentsException("Bandit Environment GameObject needs to be set!");
        }
    }

    public BanditCube GetNewBandit()
    {
        return bandits[Random.Range(0, bandits.Length)];
    }

    public override void AcademyStep()
    {
        
    }

}
