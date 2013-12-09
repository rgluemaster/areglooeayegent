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

import java.util.ArrayList;
import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

import util.Stat;
import util.Util;

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
    private double e_t;
    private double a_t;
    private double exp_decrease;
    
    //Dirichlet modelling of MDP
    private double[][][] Dirichlet; 	//Dirichlet[s][a][s']
    private double[][] DirichletSum; 	//Dirichlet[s][a]
    private Stat<Double>[][] R; 				//R[s][a]
   	private int[] pi;
   	private double[] V;
   	double VALUE_ITERATION_LIMIT = 0.00001;
    
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
		a_t = 0.8;
		exp_decrease = 0.97;
		
		//Dirichlet init
		Dirichlet = new double[nStates][nActions][nStates]; 	
	    DirichletSum = new double[nStates][nActions];
	   	pi = new int[nStates];
	   	V = new double[nStates];
    
	    R = new Stat[nStates][nActions];
	    for(int i = 0;i<nStates;i++) {
	    	for(int j = 0;j<nActions;j++) {
	    		R[i][j] = new Stat<Double>();
	    	}
	    }
    }

    public Action agent_start(Observation observation) {
        
    	//Random first action (don't know anything about environment yet)
        int newAction = actRange.getMin() + randGenerator.nextInt(nActions);
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};

        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }

    public Action agent_step(double reward, Observation observation) {
    	
    	//Update all models
    	updateQLearning(reward, observation);
    	updateDirichlet(reward, observation,VALUE_ITERATION_LIMIT);
    	
    	//Take new action
//    	int newAction = nextQLearningAction(observation);
    	int newAction = nextDirichletAction(observation);
    	
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }
    
	public void agent_end(double reward) {
		int lastState  = lastObservation.getInt(0);
    	int action = lastAction.getInt(0);
    	
    	//Update models
    	Q[lastState][action] = (1-a_t)*Q[lastState][action] + a_t*reward;
    	R[lastState][action].addObservation(reward);
    	if(reward < 0) {
    		Dirichlet[lastState][action][0] += 1;
    	} else {
    		Dirichlet[lastState][action][20] += 1;
    	}
    	DirichletSum[lastState][action] += 1;
    	valueIteration(VALUE_ITERATION_LIMIT);
    	
        lastObservation = null;
        lastAction = null;
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
    
    private void updateDirichlet(double reward, Observation observation, double limit) {
    	int lastState  = lastObservation.getInt(0);
    	int thisState  = observation.getInt(0);
    	int action = lastAction.getInt(0);
    	
    	Dirichlet[lastState][action][thisState] += 1;
    	DirichletSum[lastState][action] += 1;
    	R[lastState][action].addObservation(reward);
    	valueIteration(limit);
	}

	private void updateQLearning(double reward, Observation obs) {
    	int lastState  = lastObservation.getInt(0);
    	int thisState  = obs.getInt(0);
    	int action = lastAction.getInt(0);
    	
    	
    	//Find v(newState) = max_a Q[newState][a]
    	double Q_max  = Q[thisState][actRange.getMin()];
    	for(int a = actRange.getMin()+1;a<actRange.getMin()+nActions;a++) {
    		if(Q[thisState][a]>Q_max) {
    			Q_max = Q[thisState][a];
    		}
    	}
    	
    	//Update Q
    	Q[lastState][action] = (1-a_t)*Q[lastState][action] + a_t*(reward + y*Q_max);
    	
    	a_t *= exp_decrease;
    }
	
	private int nextDirichletAction(Observation observation) {
		return pi[observation.getInt(0)];
	}
	
	private int nextQLearningAction(Observation observation) {
    	double rand = randGenerator.nextDouble();
    	int thisState = observation.getInt(0);
    	int nextAction = 0;
    	if(rand > e_t) {
    		nextAction = randArgMax(Q[thisState]);
    	} else {
    		nextAction = actRange.getMin() + randGenerator.nextInt(nActions);
    	}
    	e_t *= exp_decrease;
    	return nextAction;
	}
	
	private int randArgMax(double[] array){
		ArrayList<Integer> maxima = new ArrayList<Integer>();
		double max = array[0];
		maxima.add(0);
		for(int i = 1;i<array.length;i++) {
			if(array[i] > max) {
				max = array[i];
				maxima.clear();
				maxima.add(i);
			} else if(array[i] == max){
				maxima.add(i);
			}
		}
		return maxima.get(randGenerator.nextInt(maxima.size()));
	}
    
    /**
	 * Value iteration using Dirichlet as a model for
	 * transition probabilities and the mean value as
	 * an estimate for the rewards.
	 */
	public void valueIteration(double limit){
		double actionValue[] = new double[nActions];
		double Vtemp[] = new double[nStates];
		
		//Initialize V
		int aBest = 0;
		double p = 0;
		for(int s = 0;s<nStates;s++){
			for(int a = 0; a<nActions;a++) {
				actionValue[a] = 0;
				for(int ss = 0;ss<nStates;ss++) {
					if(DirichletSum[s][a] > 0) {
						p = Dirichlet[s][a][ss]/DirichletSum[s][a];
					}
					actionValue[a] += p*R[s][a].getMean();
				}
			}
			aBest = randArgMax(actionValue);
			pi[s] = aBest;
			Vtemp[s] = actionValue[aBest];
		}
		
		V[0] = Double.MAX_VALUE; //Just to make the norm infinitely large.
		
		double e = Math.abs(Util.norm(V)-Util.norm(Vtemp));
			
		//Algorithm starts
		while(e > limit) {
			for(int s = 0;s<nStates;s++){
				for(int a = 0; a<nActions;a++) {
					actionValue[a] = 0;
					for(int ss = 0;ss<nStates;ss++) {
						p = 0;
						if(DirichletSum[s][a] > 0) {
							p = Dirichlet[s][a][ss]/DirichletSum[s][a];
						}
						actionValue[a] += p*(R[s][a].getMean()+y*Vtemp[ss]);
					}
				}
				aBest = randArgMax(actionValue);
				pi[s] = aBest;
				V[s] = actionValue[aBest];
			}
			
			e = Math.abs(Util.norm(V)-Util.norm(Vtemp));
			
			//Let Vtemp = V for next iteration
			Vtemp = V.clone();
		}
		
	}


}