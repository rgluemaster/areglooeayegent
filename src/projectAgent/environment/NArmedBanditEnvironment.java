package projectAgent.environment;

import java.util.Random;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.types.Reward_observation_terminal;
import org.rlcommunity.rlglue.codec.util.EnvironmentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpecVRLGLUE3;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

/**
 * @author Sebastian Anerud
 */
public class NArmedBanditEnvironment implements EnvironmentInterface {

private double[] bandits = {0.1,0.2,0.2,0.3,0.3,0.5,0.5,0.6,0.75,0.8};
private double discountFactor = 0.99;

    public String env_init() {

        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
        theTaskSpecObject.setEpisodic();
        theTaskSpecObject.setDiscountFactor(1.0d);

        //Specify that there will be an integer observation [0,nRows*nColumns-1] for the state
        theTaskSpecObject.addDiscreteObservation(new IntRange(0, 0));
        //Specify that there will be an integer action [0,3]
        theTaskSpecObject.addDiscreteAction(new IntRange(0, bandits.length-1));
        //Specify the reward range [0,1]
        theTaskSpecObject.setRewardRange(new DoubleRange(0, 1));

        theTaskSpecObject.setExtra("NArmedBanditEnvironment (Java) by Sebastian Anerud.");

        String taskSpecString = theTaskSpecObject.toTaskSpec();
        TaskSpec.checkTaskSpec(taskSpecString);

        return taskSpecString;
    }

    /**
     * Put the environment in a random state and return the appropriate observation.
     * @return
     */
    public Observation env_start() {
        Observation theObservation = new Observation(1, 0, 0);
        theObservation.setInt(0, 0);
        return theObservation;
    }

    /**
     * Make sure the action is in the appropriate range, update the state,
     * generate the new observation, reward, and whether the episode is over.
     * @param thisAction
     * @return
     */
    public Reward_observation_terminal env_step(Action thisAction) {
        /* Make sure the action is valid */
        assert (thisAction.getNumInts() == 1) : "Expecting a 1-dimensional integer action. " + thisAction.getNumInts() + "D was provided";
        assert (thisAction.getInt(0) >= 0) : "Action should be in [0,nBandits-1], " + thisAction.getInt(0) + " was provided";
        assert (thisAction.getInt(0) < bandits.length) : "Action should be in [0,nBandits-1], " + thisAction.getInt(0) + " was provided";

        Observation theObservation = new Observation(1, 0, 0);
        theObservation.setInt(0, 0);
        Reward_observation_terminal RewardObs = new Reward_observation_terminal();
        RewardObs.setObservation(theObservation);
        
        double reward = 0;
        int action = thisAction.getInt(0);
        if(Math.random()<bandits[action]) {
        	reward = 1;
        }
        RewardObs.setReward(reward);
    	RewardObs.setTerminal(Math.random()>discountFactor);
        
        return RewardObs;
    }

    public void env_cleanup() {
    }

    public String env_message(String message) {
        return "NArmedBanditEnvironment (Java) does not understand your message.";
    }

    /**
     * This is a trick we can use to make the agent easily loadable.
     * @param args
     */
    public static void main(String[] args) {
        EnvironmentLoader theLoader = new EnvironmentLoader(new NArmedBanditEnvironment());
        theLoader.run();
    }
}

