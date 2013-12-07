package projectAgent;
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
* 
*  $Revision: 676 $
*  $Date: 2009-02-08 18:15:04 -0700 (Sun, 08 Feb 2009) $
*  $Author: brian@tannerpages.com $
*  $HeadURL: http://rl-glue-ext.googlecode.com/svn/trunk/projects/codecs/Java/examples/skeleton-sample/SkeletonAgent.java $
* 
*/

import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

/**
 *
 * @author Brian Tanner (SkeletonAgent.java)
 * Edited by Sebastian Anerud and Roland Hellstrom
 */
public class AgentSmith implements AgentInterface {

    Random randGenerator = new Random();
    Action lastAction;
    Observation lastObservation;
    
    //General variables
    private TaskSpec taskSpec;
    private IntRange obsRange;
    private IntRange actRange;
    private DoubleRange rewardRange;
    private int nActions;
    private int nStates;
    private double y;

    //Q-learning variables
    private double[][] Q;
    private int[] pi;
    private double e_t;
    private double a_t;
    private double exp_decrease;
    
    public void agent_init(String taskSpecification) {
		//General init
    	taskSpec = new TaskSpec(taskSpecification);
		obsRange = taskSpec.getDiscreteObservationRange(0);
		actRange = taskSpec.getDiscreteActionRange(0);
		rewardRange = taskSpec.getRewardRange();
		y = taskSpec.getDiscountFactor();
		nStates = obsRange.getMax() - obsRange.getMin() + 1;
		nActions = actRange.getMax()-actRange.getMin() + 1;
		
		//Q-learning init
		Q = new double[nStates][nActions];
		pi = new int[nStates];
		e_t  = 0.5;
		a_t = 0.5;
		exp_decrease = 0.95;
    }

    public Action agent_start(Observation observation) {
        
    	//Random first action (don't know anything about environment yet)
        int newAction = actRange.getMin() + randGenerator.nextInt(nActions + 1);
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};

        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }

    public Action agent_step(double reward, Observation observation) {
    	
    	//Update all models
    	updateQLearning(reward, observation);
    	
    	//Take new action
    	int newAction = nextQLearningAction(observation);
    	
    	
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }

    private int nextQLearningAction(Observation observation) {
		// TODO Auto-generated method stub
		return 0;
	}

	private void updateQLearning(double reward, Observation obs) {
    	int lastState  = lastObservation.getInt(0);
    	int newState  = obs.getInt(0);
    	int action = lastAction.getInt(0);
    	
    	//Find v(newState) = max_a Q[newState][a]
    	double Q_max  = Q[newState][actRange.getMin()];
    	for(int a = actRange.getMin()+1;a<actRange.getMin()+nActions;a++) {
    		if(Q[newState][a]>Q_max) {
    			Q_max = Q[newState][a];
    		}
    	}
    	
    	//Update Q
    	Q[lastState][action] = (1-a_t)*Q[lastState][action] + a_t*(reward + y*Q_max);
    }

	public void agent_end(double reward) {
		//Do nassing
    }

    public void agent_cleanup() {
        lastAction=null;
        lastObservation=null;
    }

    public String agent_message(String message) {
        if(message.equals("what is your name?"))
            return "Agent Smith!";

	return "Mr. Anderson?";
    }
    
    /**
     * This is a trick we can use to make the agent easily loadable.
     * @param args
     */
    
    public static void main(String[] args){
     	AgentLoader theLoader=new AgentLoader(new AgentSmith());
        theLoader.run();
	}

}
