#' library(caret)
#' library(mlr3pipelines)
#' library(mlr3)
#' library(mlr3tuning)
#' library(paradox)
#'
#' # load data.
#' data(GermanCredit)
#' abt <- GermanCredit

#' Create Task for Compliance Training
#'
#' @param abt \code{data.frame}
#' @param target \code{character} name of target variable.
#'
#' @return \code{task}.
#'
#' @importFrom mlr3 TaskClassif
create_task <- function(abt, target = "Class", test_oot = FALSE,
                        label_cols = c("cpr_nr", "se_nr", "rating")) {

  # create task.
  task <- TaskClassif$new(id = "compliance_training", backend = abt, target = target)


  # what variables should be treated as 'label' cols (=not as features)?
  label_cols <- names(abt)[names(abt) %in% label_cols]
  if (length(label_cols) > 0) {
    task$set_col_role(label_cols, new_roles = "label")
  }

  # divide data into training and test.
  if (test_oot && length(unique(abt$aar) >= 0)) {
    train.idx <- abt$aar < max(abt$aar)
    task$row_roles$use <- train.idx
  }

  task

}

#' Create Compliance Graph Learner
#'
#' @importFrom mlr3pipelines mlr_pipeops Graph GraphLearner
#' @importFrom mlr3 mlr_learners
#' @importFrom mlr3filters mlr_filters
#'
#' @return \code{learner}
create_graph_learner <- function() {

  # sæt graph.
  scale <- mlr_pipeops$get("scale", id = "scale")
  xgb_learner <- mlr_pipeops$get("learner",
                                 mlr_learners$get("classif.xgboost"),
                                 id = "xgb")

  # lav graph.
  graph <- Graph$new()$
    add_pipeop(scale)$
    add_pipeop(xgb_learner)

  # forbind noder.
  graph <- graph$
    add_edge("scale", "xgb")

  # konstruér graph learner.
  GraphLearner$new(id = "Pipeline", graph)

}

#' Create Tuner for Model Training
#'
#' @param GL
#' @param validation_scheme
#'
#' @importFrom paradox ParamSet ParamDbl
#' @importFrom mlr3 mlr_resamplings mlr_measures
#' @importFrom mlr3tuning AutoTuner tnr term
#'
#' @return tuner.
create_tuner <- function(GL,
                         validation_scheme = mlr_resamplings$get("cv"),
                         measure = mlr_measures$mget("classif.acc")) {

  # sæt parametre, der skal tunes over.
  param_set <- ParamSet$new(params = list(
    ParamDbl$new("xgb.eta", lower = 0.01, upper = 0.2)
  ))

  # define tuning.
  terminator <- term("evals", n_evals = 5)
  tuner <- tnr("grid_search")

  # create auto tuner.
  AutoTuner$new(GL, validation_scheme, measure, param_set, terminator,
                tuner)

}



#
# at$store_bmr = TRUE
# at$train(task)
#
# # beregn prædiktioner.
# task$row_roles$use <- test.idx
# at$predict(task_scor)


model_configs <-
  list(
    complianceBorger = list(
      create_tuner = create_tuner,
      create_graph_learner = create_graph_learner
    )
  )


#' Title
#'
#' @param abt
#' @param col_target
#' @param col_labels
#' @param test_oot
#' @param validation_scheme
#' @param measure
#' @param model
train_model <- function(abt,
                        col_target = "Class",
                        col_labels = c("cpr_nr", "se_nr", "rating"),
                        test_oot = FALSE,
                        validation_scheme = mlr_resamplings$get("cv"),
                        measure = mlr_measures$mget("classif.acc"),
                        model = "complianceBorger") {

  # uddrag modelkonfiguration - tuner + graphLearner-funktioner.
  config <- model_configs[[model]]

  task <- create_task(abt, target = col_target, test_oot = test_oot,
                      label_cols = col_labels)

  graph_learner <- config$create_graph_learner()

  tuner <- config$create_tuner(graph_learner, validation_scheme, measure)

  # note: skal givetvis sættes andet sted.
  set.seed(1)
  tuner$train(task)

}
