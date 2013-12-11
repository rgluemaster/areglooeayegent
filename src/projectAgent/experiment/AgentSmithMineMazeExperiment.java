package projectAgent.experiment;
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
 *  $Revision: 991 $
 *  $Date: 2009-02-08 17:29:20 -0700 (Sun, 08 Feb 2009) $
 *  $Author: brian@tannerpages.com $
 *  $HeadURL: http://rl-library.googlecode.com/svn/trunk/projects/packages/examples/mines-sarsa-java/SampleExperiment.java $
 * 
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;
import org.rlcommunity.rlglue.codec.RLGlue;

/**
 * Experiment program that does some of the things that might be important when
 * running an experiment.  It runs an agent on the environment and periodically
 * asks the agent to "freeze learning": to stop updating its policy for a number
 * of episodes in order to get an estimate of the quality of what has been learned
 * so far.
 *
 * The experiment estimates statistics such as the mean and standard deviation of
 * the return gathered by the policy and writes those to a comma-separated value file
 * called results.csv.
 *
 * This experiment also shows off some other features that can be achieved easily
 * through the RL-Glue env/agent messaging system by freezing learning (described
 * above), having the environment start in specific starting states, and saving
 * and loading the agent's value function to/from a binary data file.
 * @author Brian Tanner
 */
public class AgentSmithMineMazeExperiment {

    class evaluationPoint {

        public double mean;
        public double standardDeviation;

        public evaluationPoint(double mean, double standardDeviation) {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
    }

    void printScore(int afterEpisodes, evaluationPoint theScore) {
        System.out.printf("%d\t\t%.2f\t\t%.2f\n", afterEpisodes, theScore.mean, theScore.standardDeviation);
    }

    double offlineDemo() {
        int i = 0;
        double sum = 0;
        double sum_of_squares = 0;
        double this_return = 0;
        double mean;
        double variance;
        int n = 1000;

        for (i = 1; i <= n; i++) {
            /* We use a cutoff here in case the policy is bad
            and will never end an episode */
        	RLGlue.RL_env_message("set-start-state 8 10");
            RLGlue.RL_episode(10000);
            this_return = RLGlue.RL_return();
//            if(this_return>0) {
//            	System.out.println(this_return);
//            }
            
            sum += this_return;
            sum_of_squares += this_return * this_return;
            
            if(i%((double)n / 20.0)==0) {
            	 mean = sum / (double)i;
                 variance = (sum_of_squares - (double)i * mean * mean) / ((double)i - 1.0f);
                 printScore(i, new evaluationPoint(mean, Math.sqrt(variance)));
            }
        }
        return sum;
    }

    public void runExperiment() {
        RLGlue.RL_init();
        System.out.println("Starting offline experiment\n----------------------------\n");
        System.out.println("After Episode\tMean Return\tStandard Deviation\n-------------------------------------------------------------------------");
        double totalReward = offlineDemo();

        System.out.println("-------------------------------------------------------------------------");
        System.out.println("Total reward was: " + totalReward + "\n");
        System.out.println("\nProgram Complete.");
        RLGlue.RL_cleanup();

    }

    public static void main(String[] args) {
        AgentSmithMineMazeExperiment theExperiment = new AgentSmithMineMazeExperiment();
        theExperiment.runExperiment();
    }
}
