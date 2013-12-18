package projectAgent.agent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;

import util.Stat;
import util.Util;

/**
 *
 * @author Sebastian Anerud and Roland Hellstrom
 * 
 * At the moment the algorithms "Q-learning" and "ValueIteration"
 * are not used and therefore there code is commented out to
 * enhance execution time.
 */
public class AgentSmith implements AgentInterface {

    //Aggregation variables
    private static final int LARGE_STATE_SPACE_THRESHOLD = 500;
    private static final int AGGREGATION_COUNT_THRESHOLD = 0; //Put to 0 to avoid using aggregation
    private static final boolean useTwoActions = false;
    private HashMap<Integer,Double> dirichletAlphaSNoAgg;
    private HashMap<HashKey,Double> dirichletAlphaSANoAgg;
    private HashMap<HashKey,Double> dirichletAlphaSASNoAgg;

    //General variables
    Random randGenerator = new Random();
    Action lastAction;
    Action lastLastAction;
    Action lastLastLastAction;

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
    private int episode;

    //Q-learning variables
    private double[][] Q;
    private double Q_e_t;
    private double Q_a_t;
    private double exp_decrease;
    
    //Dirichlet model of transition probability and mean to model the rewards
    private HashMap<HashKey,Double> dirichletAlphaSAS;
    private HashMap<HashKey,Double> dirichletAlphaSA;
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
    private boolean manyStates;

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

//        String responseMessage = RLGlue.RL_env_message("what is your name?");
//        System.out.println("Environment responded to \"what is your name?\" with: " + responseMessage);
        //String responseMessage = RLGlue.RL_env_message("what is your name?");
        ////System.out.println("Environment responded to \"what is your name?\" with: " + responseMessage);

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

        if(nStates > LARGE_STATE_SPACE_THRESHOLD){
            manyStates = true;
        }


		nActions = actRange.getMax()-actRange.getMin() + 1;
        System.out.println("Number of actions: " + nActions);

        System.out.println("______________________________________________________________ \n");

        time = 0;
		initDirichlet();
//		initQLearning();
//		initVI();
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
    		newAction = actRange.getMin() + randGenerator.nextInt(nActions);
    	} else {
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
//    	updateQLearning(reward, observation);
//    	updateVI();
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
                Observation obs2 = observation.duplicate();
                if(getAlpha(obs2.getInt(0)) < AGGREGATION_COUNT_THRESHOLD){
                    if(lastLastLastAction != null && useTwoActions){
                        obs2.setInt(0, calculateSecondAggState(lastAction, lastLastAction));
                    }else{
                        obs2.setInt(0, calculateAggState(lastAction));
                    }
                }
//    			newAction = randGenerator.nextInt(9);
    			newAction = nextGSVIAction(obs2);
    			algorithm = "GSVI";
    		}
    	}

        Action returnAction = new Action();
        returnAction.intArray = new int[]{actRange.getMin() + newAction};

        if(lastLastAction != null && useTwoActions){
            lastLastLastAction = lastLastAction.duplicate();
        }
        lastLastAction = lastAction.duplicate();
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
        System.out.println(dirichletAlphaS.size());
        System.out.println("End. Reward: " + reward + ". Episode reward: " + episodeReward + ".");
        System.out.println("______________________________________________________________ \n");
    	//Update models
		endDirichlet(reward);
//    	endQLearning(reward);
//    	endVI();
    	endGSVI();
    	endUCB1(reward);


        time = 0;
        totalReward += episodeReward;
        episodeReward = 0;

        lastObservation = null;
        lastLastLastAction = null;
        lastLastAction = null;
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


//        for(int i = obsRange.getMax(); i < GSVI_Q.length; i++){
//            for(int j = 0; j < GSVI_Q[0].length; j++){
//                System.out.println("Q: i:" + i  + " j:" + j + " value: " + GSVI_Q[i][j]);
//                Action a = new Action(1, 0);
//                a.setInt(0, j);
//                System.out.println("Action aggstate: " + calculateAggState(a));
//            }
//        }
//
//        for(int i = obsRange.getMax(); i < rewards.length; i++){
//            for(int j = 0; j < rewards[0].length; j++){
//                System.out.println("Rewards: i:" + i  + " j:" + j + " value: " + rewards[i][j] + " sample size: " + rewards[i][j].getSampleSize());
//                Action a = new Action(1, 0);
//                a.setInt(0, j);
//                System.out.println("Action aggstate: " + calculateAggState(a) + "\n");
//            }
//        }



        System.out.println("______________________________________________________________");
		System.out.println("Agent clean up called. Total reward: " + totalReward + ".");
		System.out.println("Number of states in HashMap: " + dirichletAlphaS.size());
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
        dirichletAlphaSASNoAgg = new HashMap<HashKey,Double>();
        dirichletAlphaSANoAgg = new HashMap<HashKey,Double>();
        dirichletAlphaSNoAgg = new HashMap<Integer,Double>();
        dirichletAlphaSAS = new HashMap<HashKey,Double>();
    	dirichletAlphaSA = new HashMap<HashKey,Double>(); 	
    	dirichletAlphaS = new HashMap<Integer,Double>(); 	
	    rewards = new Stat[nStates + nActions*nActions + nActions][nActions];
	    for(int i = 0;i<rewards.length;i++) {
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
		exp_decrease = 0.99;
	}

	private void initVI() {
	   	VI_pi = new int[nStates];
	   	VI_V = new double[nStates];
	}
	
	private void initGSVI() {
		GSVI_pi = new int[nStates + nActions*nActions + nActions]; //TODO: Magical numbers..
		GSVI_V = new double[nStates + nActions*nActions + nActions];
	   	GSVI_Q = new double[nStates + nActions*nActions + nActions][nActions];
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

        incrementAlphaNoAgg(lastState, action, thisState);
        incrementAlpha(lastState,action,thisState); // Must be called after incrementAlphaNoAgg(lastState, action, thisState);.

        if(lastLastAction != null){
            rewards[calculateAggState(lastLastAction)][action] .addObservation(reward); //Note that both the aggregated and disaggregate model are always updated
        }

        if(lastLastLastAction != null && useTwoActions){
            rewards[calculateSecondAggState(lastLastAction, lastLastLastAction)][action] .addObservation(reward);
        }


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
            //e = 1.0/10.0;
            e = Math.pow(n, -2);
            //e = 1.0/n;
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
				//if(alphaSum>0){
				//	learningRate = 1/Math.log(Math.exp(1)+alphaSum-1);
				//}
				actionValue[a] = Qtemp[s][a]*(1-learningRate) + learningRate*(rewards[s][a].getMean());
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
	
	private double getAlpha(int s, int a, int ss){
		HashKey hashKey = new HashKey(s,a,ss);
		Double value = dirichletAlphaSAS.get(hashKey);
		if(value == null) {
			return 0;
		} else {
			return value;
		}
	}
	
	private double getAlpha(int s, int a){
		HashKey hashKey = new HashKey(s,a);
		Double value = dirichletAlphaSA.get(hashKey);
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
	
	private void incrementAlpha(int lastState, int a, int thisState){
        Double value = dirichletAlphaSNoAgg.get(lastState);
        if(value >= AGGREGATION_COUNT_THRESHOLD){
            dirichletAlphaS.put(lastState, value);
        }
        if(lastLastAction != null){
            int lastAggState = calculateAggState(lastLastAction); //Note that we always update the aggstate
            Double valueAgg = dirichletAlphaS.get(lastAggState);
            if(valueAgg != null) {
                dirichletAlphaS.put(lastAggState, valueAgg + 1.0);
            } else {
                dirichletAlphaS.put(lastAggState, 1.0);
            }
        }
        if(lastLastLastAction != null && useTwoActions){
            int lastSecondAggState = calculateSecondAggState(lastLastAction, lastLastLastAction); //Note that we always update the aggstate
            Double valueSecondAgg = dirichletAlphaS.get(lastSecondAggState);
            if(valueSecondAgg != null) {
                dirichletAlphaS.put(lastSecondAggState, valueSecondAgg + 1.0);
            } else {
                dirichletAlphaS.put(lastSecondAggState, 1.0);
            }
        }



        HashKey hashKey = new HashKey(lastState,a);
        value = dirichletAlphaSANoAgg.get(hashKey);
        if(value >= AGGREGATION_COUNT_THRESHOLD){
            dirichletAlphaSA.put(hashKey, value);
        }
        if(lastLastAction != null){
            int lastAggState = calculateAggState(lastLastAction);
            Double valueAgg = dirichletAlphaSA.get(lastAggState);
            hashKey = new HashKey(lastAggState, a);
            if(valueAgg != null) {
                dirichletAlphaSA.put(hashKey, valueAgg + 1.0);
            } else {
                dirichletAlphaSA.put(hashKey, 1.0);
            }
        }
        if(lastLastLastAction != null && useTwoActions){
            int lastSecondAggState = calculateSecondAggState(lastLastAction, lastLastLastAction);
            Double valueSecondAgg = dirichletAlphaSA.get(lastSecondAggState);
            hashKey = new HashKey(lastSecondAggState, a);
            if(valueSecondAgg != null) {
                dirichletAlphaSA.put(hashKey, valueSecondAgg + 1.0);
            } else {
                dirichletAlphaSA.put(hashKey, 1.0);
            }
        }

        hashKey = new HashKey(lastState,a,thisState);
        value = dirichletAlphaSASNoAgg.get(hashKey);
        if(value >= AGGREGATION_COUNT_THRESHOLD){
            dirichletAlphaSAS.put(hashKey, value);
        }
        if(lastLastAction != null){
            int lastAggState = calculateAggState(lastLastAction);
            int aggState = calculateAggState(lastAction);
            Double valueAgg = dirichletAlphaSAS.get(lastAggState);
            hashKey = new HashKey(lastAggState, a, aggState);
            if(valueAgg != null) {
                dirichletAlphaSAS.put(hashKey, valueAgg + 1.0);
            } else {
                dirichletAlphaSAS.put(hashKey, 1.0);
            }
        }
        if(lastLastLastAction != null && useTwoActions){
            int lastSecondAggState = calculateSecondAggState(lastLastAction, lastLastLastAction);
            int secondAggState = calculateSecondAggState(lastAction, lastLastAction);
            Double valueSecondAgg = dirichletAlphaSAS.get(lastSecondAggState);
            hashKey = new HashKey(lastSecondAggState, a, secondAggState);
            if(valueSecondAgg != null) {
                dirichletAlphaSAS.put(hashKey, valueSecondAgg + 1.0);
            } else {
                dirichletAlphaSAS.put(hashKey, 1.0);
            }
        }
	}

    private int calculateAggState(Action lastAction) {
        return ((obsRange.getMax() + lastAction.getInt(0)) % nActions ) + obsRange.getMax() + 1;
    }

    private int calculateSecondAggState(Action lastAction, Action lastLastAction) {
        return ((obsRange.getMax() + lastLastAction.getInt(0)) % nActions )*nActions + ((obsRange.getMax() + lastAction.getInt(0)) % nActions ) + nActions + obsRange.getMax() + 1;
    }

    //alphanoagg is needed!
    private void incrementAlphaNoAgg(int s, int a, int ss){
        HashKey hashKey = new HashKey(s,a,ss);
        Double value = dirichletAlphaSASNoAgg.get(hashKey);
        if(value != null) {
            dirichletAlphaSASNoAgg.put(hashKey,value+1);
        } else {
            dirichletAlphaSASNoAgg.put(hashKey,1.0);
        }

        hashKey = new HashKey(s,a);
        value = dirichletAlphaSANoAgg.get(hashKey);
        if(value != null) {
            dirichletAlphaSANoAgg.put(hashKey,value+1);
        } else {
            dirichletAlphaSANoAgg.put(hashKey,1.0);
        }

        value = dirichletAlphaSNoAgg.get(s);
        if(value != null) {
            dirichletAlphaSNoAgg.put(s,value+1);
        } else {
            dirichletAlphaSNoAgg.put(s,1.0);
        }
    }
	
	public class HashKey{
		private int[] key;
		
		public HashKey(int s, int a, int ss){
			key = new int[3];
			key[0] = s;
			key[1] = a;
			key[2] = ss;
		}
		
		public HashKey(int s, int a){
			key = new int[2];
			key[0] = s;
			key[1] = a;
		}


		@Override
		public boolean equals(Object o){
			if(!(o instanceof HashKey)) {
				return false;
			}
			HashKey hk = (HashKey) o;
			if(hk.key.length != this.key.length) {
				return false;
			}
			
			for(int i = 0;i<this.key.length;i++) {
				if(hk.key[i] != this.key[i]) {
					return false;
				}
			}
			return true;
		}
		
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			for(int i = 0; i< key.length;i++) {
				sb.append(key[i] + " ");
			}
			return sb.toString();
		}
		
		@Override
		public int hashCode() {
			int[] prime = {5,7,13};
			int hash = 0;
			for(int i = 0; i<key.length;i++){
				hash += key[i]*prime[i];
			}
			return hash;
		}
	}
}
