<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class FinancialData extends Model
{
    /**
     * Get sample income data as JSON.
     *
     * @param mixed $appid
     * @param mixed $accno
     * @param mixed $primaryIncome
     * @param mixed $otherIncome
     * @param mixed $shareRent
     * @param mixed $investIncome
     * @param mixed $partnerSalary
     * @param mixed $childSupport
     * @param mixed $govnSubsidy
     * @return string JSON encoded income data
     */
    public static function getIncome($appid, $accno, $primaryIncome, $otherIncome, $shareRent, $investIncome, $partnerSalary, $childSupport, $govnSubsidy)
    {
        $incomeData = [
            [
                'appid'         => $appid,
                'accno'         => '122323',
                'primaryIncome' => 24003,
                'otherincome'   => 2342,
                'shareRent'     => 34232,
                'InvestIncom'   => 324,
                'partnersalary' => 3243242,
                'childsupport'  => 32432,
                'govnsubsidy'   => 23432,
            ],
            [
                'appid'         => $appid,
                'accno'         => '343543',
                'primaryIncome' => 4353,
                'otherincome'   => 2342,
                'shareRent'     => 345,
                'InvestIncom'   => 324,
                'partnersalary' => 3243242,
                'childsupport'  => 32432,
                'govnsubsidy'   => 23432,
            ],
        ];

        return json_encode($incomeData);
    }

    /**
     * Get sample consumer expense data as JSON.
     *
     * @param mixed $appid
     * @param mixed $accno
     * @param mixed $mortage
     * @param mixed $a
     * @param mixed $b
     * @return string JSON encoded consumer expense data
     */
    public static function getConsumerExpense($appid, $accno, $mortage, $foodEntmt, $travelExp, $homeUtility, $insurance, $personalExp, $smallLoans, $otherLoans  )
    {
        $expenseData = [
            [
                'appid'   => $appid,
                'accno'   => 92103912,
                'mortage' => 3432,
                'foodEntmt'       => 4234,
                'travelExp'       => 2342,
                'homeUtility'     => 435,
                'insurance'       => 32423,
                'personalExp'     => 2365464,
                'smallLoans'      => 234534,
                'otherLoans'      => 233424,

            ],
            [
                'appid'   => $appid,
                'accno'   => 3453,
                'mortage' => 324,
                'foodEntmt'       => 2342,
                'travelExp'       => 2342,
                'homeUtility'     => 437685,
                'insurance'       => 768768,
                'personalExp'     => 3254,
                'smallLoans'      => 234543534,
                'otherLoans'      => 546,
            ],
        ];

        return json_encode($expenseData);
    }
}
