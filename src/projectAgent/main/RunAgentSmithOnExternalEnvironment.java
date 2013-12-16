package projectAgent.main;

import org.rlcommunity.rlglue.codec.util.AgentLoader;
import projectAgent.agent.AgentSmith;
import projectAgent.experiment.AgentSmithSkeletonExperiment;

public class RunAgentSmithOnExternalEnvironment {

    public static void main(String[] args){

        AgentLoader theAgentLoader=new AgentLoader(new AgentSmith());

        Thread agentThread=new Thread(theAgentLoader);

        agentThread.start();

        AgentSmithSkeletonExperiment.main(args);
        System.out.println("RunAll Complete");

        System.exit(1);
    }
}



