package projectAgent.agent;
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

	//General variables
    Random randGenerator = new Random();
    Action lastAction;
    Observation lastObservation;
    private TaskSpec taskSpec;
    private IntRange obsRange;
    private IntRange actRange;
    private DoubleRange rewardRange;
    private int nActions;
    private int nStates;
    private double discountFactor;
    private boolean isFirstEpisode= true;

    //Q-learning variables
    private double[][] Q;
    private double Q_e_t;
    private double Q_a_t;
    private double exp_decrease;
    
    //Dirichlet model of transition probability and mean to model the rewards
    private double[][][] dirichletAlpha; 	//Dirichlet[s][a][s']
    private double[][] dirichletAlphaSum; 	//Dirichlet[s][a]
    private Stat<Double>[][] rewards; 				//R[s][a]
    
    //Value Iteration
   	private int[] VI_pi;
   	private double[] VI_V;
   	double VALUE_ITERATION_LIMIT = 0.00001;
   	
   	//Generalized Stochastic value iteration
   	private int[] GSVI_pi;
   	private double[] GSVI_V;
   	private double[][] GSVI_Q;
   	
   	//UCB1 algorithm for bandit problems
    private Stat<Double>[] UCB_R;
    private double time;
    
    /**
	 * 
	 * 
	 * --------------- agent_init ---------------
	 * 
	 * 
	 */
    
    public void agent_init(String taskSpecification) {
		//General init
    	taskSpec = new TaskSpec(taskSpecification);
		obsRange = taskSpec.getDiscreteObservationRange(0);
		actRange = taskSpec.getDiscreteActionRange(0);
		rewardRange = taskSpec.getRewardRange();
		discountFactor = taskSpec.getDiscountFactor();
		nStates = obsRange.getMax() - obsRange.getMin() + 1;
		nActions = actRange.getMax()-actRange.getMin() + 1;
		
		initDirichlet();
		initQLearning();
		initVI();
		initGSVI();
	    initUCB1();
    }
    
    /**
	 * 
	 * 
	 * --------------- agent_start ---------------
	 * 
	 * 
	 */
    
	public Action agent_start(Observation observation) {
        
    	int newAction = 0;
    	if(isFirstEpisode) {
    		//First episode -> take random action
    		newAction = actRange.getMin() + randGenerator.nextInt(nActions);
    	} else {
    		//Take action
    		newAction = nextGSVIAction(observation);
    	}
        
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};

        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();
        isFirstEpisode = false;
        return returnAction;
    }

	/**
	 * 
	 * 
	 * --------------- agent_step ---------------
	 * 
	 * 
	 */
	
    public Action agent_step(double reward, Observation observation) {
    	time++;
    	//Update all models and algorithms
    	updateDirichlet(reward, observation);
    	updateQLearning(reward, observation);
    	updateVI();
    	updateGSVI();
    	updateUCB1(reward);
    	
    	//Take new action
    	int newAction = 0;
//    	newAction = nextQLearningAction(observation);
//    	newAction = nextVIAction(observation);
    	newAction = nextGSVIAction(observation);
//    	newAction = nextUCB1Action();
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};
        
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        return returnAction;
    }
    
    /**
	 * 
	 * 
	 * --------------- agent_end ---------------
	 * 
	 * 
	 */
    
	public void agent_end(double reward) {
    	//Update models
		endDirichlet(reward);
    	endQLearning(reward);
    	endVI();
    	endGSVI();
    	endUCB1(reward);
    	
        lastObservation = null;
        lastAction = null;
    }
	
	/**
	 * 
	 * 
	 * --------------- agent_cleanup ---------------
	 * 
	 * 
	 */
	
	public void agent_cleanup() {
		System.out.println("Clean up called");
        lastAction=null;
        lastObservation=null;
    }

	/**
	 * 
	 * 
	 * --------------- agent_message ---------------
	 * 
	 * 
	 */
	
    public String agent_message(String message) {
        if(message.equals("what is your name?"))
            return "Agent Smith!";

	return "Mr. Anderson?";
    }
    
    /**
	 * 
	 * 
	 * --------------- main ---------------
	 * 
	 * 
	 */
    
    public static void main(String[] args){
     	AgentLoader theLoader=new AgentLoader(new AgentSmith());
        theLoader.run();
	}
    
    /**
     * 
     * 
	 * --------------- Init methods ---------------
	 * 
	 * 
	 */
    
    private void initDirichlet() {
    	dirichletAlpha = new double[nStates][nActions][nStates]; 	
	    dirichletAlphaSum = new double[nStates][nActions];
	    rewards = new Stat[nStates][nActions];
	    for(int i = 0;i<nStates;i++) {
	    	for(int j = 0;j<nActions;j++) {
	    		rewards[i][j] = new Stat<Double>();
	    	}
	    }
	}
	
	private void initQLearning() {
    	Q = new double[nStates][nActions];
		VI_pi = new int[nStates];
		Q_e_t  = 0.5;
		Q_a_t = 0.8;
		exp_decrease = 0.97;
	}

	private void initVI() {
	   	VI_pi = new int[nStates];
	   	VI_V = new double[nStates];
	}
	
	private void initGSVI() {
		GSVI_pi = new int[nStates];
		GSVI_V = new double[nStates];
	   	GSVI_Q = new double[nStates][nActions];
	}

	private void initUCB1() {
		UCB_R = new Stat[nActions];
		for(int j = 0;j<nActions;j++) {
    		UCB_R[j] = new Stat<Double>();
    		UCB_R[j].addObservation(rewardRange.getMax());
	    }
		time = 0;
	}
	
	/**
	 * 
	 * 
	 * --------------- End methods ---------------
	 * 
	 * 
	 */
	
	private void endDirichlet(double reward) {
		int lastState  = lastObservation.getInt(0);
    	int action = lastAction.getInt(0);
    	rewards[lastState][action].addObservation(reward);
	}

    private void endUCB1(double reward) {
    	updateUCB1(reward);
	}

	private void endVI() {
    	valueIteration(VALUE_ITERATION_LIMIT);
	}
	
	private void endGSVI() {
		generalizedStochasticValueIteration();
	}

	private void endQLearning(double reward) {
		int lastState  = lastObservation.getInt(0);
    	int action = lastAction.getInt(0);
    	Q[lastState][action] = (1-Q_a_t)*Q[lastState][action] + Q_a_t*reward;
	}

	/**
	 * 
	 * 
	 * --------------- Update methods ---------------
	 * 
	 * 
	 */
    
	
	private void updateDirichlet(double reward, Observation observation) {
		int lastState  = lastObservation.getInt(0);
    	int thisState  = observation.getInt(0);
    	int action = lastAction.getInt(0);
		
    	dirichletAlpha[lastState][action][thisState] += 1;
    	dirichletAlphaSum[lastState][action] += 1;
    	rewards[lastState][action].addObservation(reward);
	}

    private void updateVI() {
    	valueIteration(VALUE_ITERATION_LIMIT);
	}
    
    private void updateGSVI() {
    	generalizedStochasticValueIteration();
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
    	Q[lastState][action] = (1-Q_a_t)*Q[lastState][action] + Q_a_t*(reward + discountFactor*Q_max);
    	
    	Q_a_t *= exp_decrease;
    }
	
	private void updateUCB1(double reward) {
		int action = lastAction.getInt(0);
		UCB_R[action].addObservation(reward);
	}
	
	/**
	 * 
	 * 
	 * --------------- Methods for taking actions ---------------
	 * 
	 * 
	 */
	
	private int nextVIAction(Observation observation) {
		int thisState = observation.getInt(0);
		double e = 1;
		double n = Util.arraySum(dirichletAlphaSum[thisState]);
		if(n>0) {
			e = 1/Math.pow(n, 2.0/3.0);
		}
		if(randGenerator.nextDouble() < e) {
			return actRange.getMin() + randGenerator.nextInt(nActions);
		} else {
			return VI_pi[thisState];
		}
	}
	
	private int nextGSVIAction(Observation observation) {
		int thisState = observation.getInt(0);
		double e = 1;
		double n = Util.arraySum(dirichletAlphaSum[thisState]);
		if(n>0) {
			e = 1/Math.pow(n, 2.0/3.0);
		}
		if(randGenerator.nextDouble() < e) {
			return actRange.getMin() + randGenerator.nextInt(nActions);
		} else {
			return GSVI_pi[thisState];
		}
		
	}
	
	private int nextQLearningAction(Observation observation) {
    	double rand = randGenerator.nextDouble();
    	int thisState = observation.getInt(0);
    	int nextAction = 0;
    	if(rand > Q_e_t) {
    		nextAction = randArgMax(Q[thisState]);
    	} else {
    		nextAction = actRange.getMin() + randGenerator.nextInt(nActions);
    	}
    	Q_e_t *= exp_decrease;
    	return nextAction;
	}
	
	private int nextUCB1Action() {
		double u[] = new double[nActions];
		for(int i = 0; i<nActions;i++) {
			double n = UCB_R[i].getSampleSize();
			double mu = UCB_R[i].getMean();
			u[i] = mu + Math.sqrt(2*Math.log(time)/n);
		}
		return randArgMax(u);
	}
	
	/**
	 * 
	 * 
	 * --------------- Other methods ---------------
	 * 
	 * 
	 */
	
	/**
	 * Returns the maximum argument of the inputed array. If there are more than
	 * one maximum argument, one of them is chosen at random.
	 * @param array the array to find the maximum argument in.
	 * @return the maximum argument (ties are chosen at random). 
	 */
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
	 * @param limit the limit for when to stop the value iteration 
	 * if(abs(|V_old|-|V_new|) < limit) then stop.
	 * 
	 */
	private void valueIteration(double limit){
		double actionValue[] = new double[nActions];
		double Vtemp[] = new double[nStates];
		
		//Initialize V
		int aBest = 0;
		double p = 0;
		for(int s = 0;s<nStates;s++){
			for(int a = 0; a<nActions;a++) {
				actionValue[a] = rewards[s][a].getMean();
			}
			aBest = randArgMax(actionValue);
			VI_pi[s] = aBest;
			Vtemp[s] = actionValue[aBest];
		}
		
		VI_V[0] = Double.MAX_VALUE; //Just to make the norm infinitely large.
		
		double change = Math.abs(Util.norm(VI_V)-Util.norm(Vtemp));
			
		//Iterate values while change > limit
		while(change > limit) {
			for(int s = 0;s<nStates;s++){
				for(int a = 0; a<nActions;a++) {
					actionValue[a] = rewards[s][a].getMean();
					for(int ss = 0;ss<nStates;ss++) {
						p = 0;
						if(dirichletAlphaSum[s][a] > 0) {
							p = dirichletAlpha[s][a][ss]/dirichletAlphaSum[s][a];
						}
						actionValue[a] += p*discountFactor*Vtemp[ss];
					}
				}
				
				aBest = randArgMax(actionValue);
				VI_pi[s] = aBest;
				VI_V[s] = actionValue[aBest];
			}
			
			change = Math.abs(Util.norm(VI_V)-Util.norm(Vtemp));
			Vtemp = VI_V.clone();
		}
		
	}

	private void generalizedStochasticValueIteration() {
		double[][] Qtemp = GSVI_Q.clone();
		double[] Vtemp = GSVI_V.clone();
		
		for(int s = 0;s<nStates;s++){
			double[] actionValue = new double[nActions];
			for(int a = 0; a<nActions;a++) {
				double learningRate = 1;
				if(dirichletAlphaSum[s][a]>0){
					learningRate = 1/Math.log(Math.exp(1)+dirichletAlphaSum[s][a]);
				}
				actionValue[a] = Qtemp[s][a]*(1-learningRate) + learningRate*rewards[s][a].getMean();
				for(int ss = 0;ss<nStates;ss++) {
					double p = 0;
					if(dirichletAlphaSum[s][a] > 0) {
						p = dirichletAlpha[s][a][ss]/dirichletAlphaSum[s][a];
					}
					actionValue[a] += learningRate*discountFactor*p*Vtemp[ss];
				}
			}
			GSVI_Q[s] = actionValue;
			
			int aBest = randArgMax(actionValue);
			GSVI_pi[s] = aBest;
			GSVI_V[s] = actionValue[aBest];
		}
 	}
}
