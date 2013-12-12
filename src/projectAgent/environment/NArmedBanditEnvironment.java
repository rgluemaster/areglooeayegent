package projectAgent.environment;
/*
 * Copyright 2008 Brian Tanner
 * http://rl-glue-ext.googlecode.com/
 * brian@tannerpages.com
 * http://brian.tannerpages.com
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.

 *  $Revision: 998 $
 *  $Date: 2009-02-08 20:21:50 -0700 (Sun, 08 Feb 2009) $
 *  $Author: brian@tannerpages.com $
 *  $HeadURL: http://rl-library.googlecode.com/svn/trunk/projects/packages/examples/mines-sarsa-java/SampleMinesEnvironment.java $

 *
 */

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
 * This code is adapted from the Mines.cpp code written by Adam White
 * for earlier versions of RL-Glue.	
 *
 * See the RL-Library page: 
 * http://library.rl-community.org/environments/mines
 *
 *	This is a very simple discrete-state, episodic grid world that has
 *	exploding mines in it.  If the agent steps on a mine, the episode
 *	ends with a large negative reward.
 *
 *	The reward per step is -1, with +10 for exiting the game in goal 1, +100 for
 *  exiting the game in goal 2 and -100 for stepping on a mine.
 *
 * @author Brian Tanner then modified by Sebastian Anerud
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

