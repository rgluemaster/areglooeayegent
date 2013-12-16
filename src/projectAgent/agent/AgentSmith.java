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
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

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
    private double totalReward;
    private double episodeReward;
    private double time;

    //Q-learning variables
    private double[][] Q;
    private double Q_e_t;
    private double Q_a_t;
    private double exp_decrease;
    
    //Dirichlet model of transition probability and mean to model the rewards
    private HashMap<String,Double> dirichletAlphaSAS;
    private HashMap<String,Double> dirichletAlphaSA;
    private HashMap<Integer,Double> dirichletAlphaS; 	
    private Stat<Double>[][] rewards; 				
    
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
    private double UCB_time;
    
    /**
	 * 
	 * 
	 * --------------- agent_init ---------------
	 * 
	 * 
	 */
    
    public void agent_init(String taskSpecification) {
        System.out.println("______________________________________________________________");
        System.out.println("Agent init called");
		//General init
    	taskSpec = new TaskSpec(taskSpecification);
		obsRange = taskSpec.getDiscreteObservationRange(0);
        System.out.println("Observation range: " + obsRange);
		actRange = taskSpec.getDiscreteActionRange(0);
        System.out.println("Action range: " + actRange);
		rewardRange = taskSpec.getRewardRange();
        System.out.println("Reward range: " + rewardRange);
		discountFactor = taskSpec.getDiscountFactor();
        System.out.println("Discount factor: " + discountFactor);
		nStates = obsRange.getMax() - obsRange.getMin() + 1;
        System.out.println("Number of states: " + nStates);
		nActions = actRange.getMax()-actRange.getMin() + 1;
        System.out.println("Number of actions: " + nActions);

        System.out.println("______________________________________________________________ \n");
		
        time = 0;
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
        System.out.println("______________________________________________________________");
        System.out.println("Start-observation: " + observation);
        
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

        System.out.println("Taking action " +returnAction + "\n");
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
        System.out.println("\n Step. Time: " +time+ ". Reward: " + reward + ". Observation: " + observation.getInt(0) + ".");
    	//Update all models and algorithms
    	updateDirichlet(reward, observation);
    	updateQLearning(reward, observation);
    	updateVI();
    	updateGSVI();
    	updateUCB1(reward);
    	
    	//Take new action
    	int newAction = actRange.getMin();
    	String algorithm = "";
    	if(nStates == 1) {
    		newAction = nextUCB1Action();
    		algorithm = "UCB1";
    	} else {
    		if(Math.random()>1) {
    			newAction = nextVIAction(observation);
    			algorithm = "VI";
    		} else{
    			newAction = nextGSVIAction(observation);
    			algorithm = "GSVI";
    		}
    	}
    	
        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};
        
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();

        episodeReward += reward;

        System.out.println("Taking action " +returnAction.getInt(0) + " according to " + algorithm + "-algorithm.");
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
        episodeReward += reward;
        System.out.println("End. Reward: " + reward + ". Episode reward: " + episodeReward + ".");
        System.out.println("______________________________________________________________ \n");
    	//Update models
		endDirichlet(reward);
    	endQLearning(reward);
    	endVI();
    	endGSVI();
    	endUCB1(reward);


        time = 0;
        totalReward += episodeReward;
        episodeReward = 0;

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
        System.out.println("______________________________________________________________");
		System.out.println("Agent clean up called. Total reward: " + totalReward + ".");
		if(nStates==1){
			StringBuilder sb = new StringBuilder();
			sb.append("Action proportion = (");
			for(int i = 0; i<nActions-1;i++) {
				sb.append(Util.roundNDecimals(getAlpha(0, i)/getAlpha(0),3) + ",");
			}
			sb.append(Util.roundNDecimals(getAlpha(0, nActions-1)/getAlpha(0),3) + ")");
			System.out.println(sb);
	        System.out.println("______________________________________________________________ \n");
		}
        totalReward = 0;
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
        System.out.println("Message called with message: " + message);
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
    	dirichletAlphaSAS = new HashMap<String,Double>(); 	
    	dirichletAlphaSA = new HashMap<String,Double>(); 	
    	dirichletAlphaS = new HashMap<Integer,Double>(); 	
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
		UCB_time = 0;
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
		if(nStates!=1) {
    		valueIteration(VALUE_ITERATION_LIMIT);
    	}
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
		
    	incrementAlpha(lastState,action,thisState);
    	incrementAlpha(lastState,action);
    	incrementAlpha(lastState);
    	
    	rewards[lastState][action].addObservation(reward);
	}

    private void updateVI() {
    	if(nStates!=1) {
    		valueIteration(VALUE_ITERATION_LIMIT);
    	}
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
		UCB_time++;
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
		double n = getAlpha(thisState);
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
		double n = getAlpha(thisState);
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
			u[i] = mu + Math.sqrt(2*Math.log(UCB_time)/n);
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
		
		Set<Integer> visibleStates = dirichletAlphaS.keySet();
		Iterator<Integer> visibleStatesIterator = visibleStates.iterator();
		//Initialize V
		int aBest = 0;
		double p = 0;
		while(visibleStatesIterator.hasNext()){
			int s = visibleStatesIterator.next();
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
			visibleStatesIterator = visibleStates.iterator();
			while(visibleStatesIterator.hasNext()){
				int s = visibleStatesIterator.next();
				for(int a = 0; a<nActions;a++) {
					actionValue[a] = rewards[s][a].getMean();
					Iterator<Integer> nextStatesIterator = visibleStates.iterator();
					while(nextStatesIterator.hasNext()){
						int ss = nextStatesIterator.next();
						p = 0;
						double alphaSum = getAlpha(s, a);
						if(alphaSum > 0) {
							p = getAlpha(s, a, ss)/alphaSum;
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
		
		Set<Integer> visibleStates = dirichletAlphaS.keySet();
		Iterator<Integer> visibleStatesIterator = visibleStates.iterator();
		
		while(visibleStatesIterator.hasNext()){
			int s = visibleStatesIterator.next();
			double[] actionValue = new double[nActions];
			for(int a = 0; a<nActions;a++) {
				double learningRate = 1;
				double alphaSum = getAlpha(s, a);
				if(alphaSum>0){
					learningRate = 1/Math.log(Math.exp(1)+alphaSum);
				}
				actionValue[a] = Qtemp[s][a]*(1-learningRate) + learningRate*rewards[s][a].getMean();
				Iterator<Integer> nextStatesIterator = visibleStates.iterator();
				while(nextStatesIterator.hasNext()){
					int ss = nextStatesIterator.next();
					double p = 0;
					if(alphaSum > 0) {
						p = getAlpha(s, a, ss)/alphaSum;
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
	
	private String toHashKey(int s, int a, int ss) {
		return s + " " + a + " " + " " + ss;
	}
	
	private String toHashKey(int s, int a) {
		return s + " " + a + " ";
	}
	
	private String toHashKey(int s) {
		return "" + s;
	}
	
	private double getAlpha(int s, int a, int ss){
		String key = toHashKey(s, a, ss);
		Double value = dirichletAlphaSAS.get(key);
		if(value == null) {
			return 0;
		} else {
			return value;
		}
	}
	
	private double getAlpha(int s, int a){
		String key = toHashKey(s, a);
		Double value = dirichletAlphaSA.get(key);
		if(value == null) {
			return 0;
		} else {
			return value;
		}
	}
	
	private double getAlpha(int s){
		Double value = dirichletAlphaS.get(s);
		if(value == null) {
			return 0;
		} else {
			return value;
		}
	}
	
	private void incrementAlpha(int s, int a, int ss){
		String hashKey = toHashKey(s,a,ss);
    	Double value = dirichletAlphaSAS.get(hashKey);
    	if(value != null) {
    		dirichletAlphaSAS.put(hashKey,value+1);
    	} else {
    		dirichletAlphaSAS.put(hashKey,1.0);
    	}
    	
	}
	
	private void incrementAlpha(int s, int a){
		String hashKey = toHashKey(s,a);
    	Double value = dirichletAlphaSA.get(hashKey);
    	if(value != null) {
    		dirichletAlphaSA.put(hashKey,value+1);
    	} else {
    		dirichletAlphaSA.put(hashKey,1.0);
    	}
	}

	private void incrementAlpha(int s){
    	Double value = dirichletAlphaS.get(s);
    	if(value != null) {
    		dirichletAlphaS.put(s,value+1);
    	} else {
    		dirichletAlphaS.put(s,1.0);
    	}
	}
}
